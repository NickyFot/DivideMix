from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from PreResNet import *
from sklearn.mixture import GaussianMixture

import dataloader_AffectNet as dataloader

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--noise_mode', default='sym')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=2, type=int)
parser.add_argument('--data_path', default='', type=str, help='path to dataset')
parser.add_argument('--dataset', default='affectnet', type=str)
parser.add_argument('--regression', dest='regression', action='store_true',
                    help="train on regression task, using the dimensional A/V model")
parser.set_defaults(regression=False)
parser.add_argument('--cls', dest='cls', action='store_true', help="train on classification task")
parser.set_defaults(cls=False)
parser.add_argument('--var', dest='var', action='store_true', help="train on kldiv task")
parser.set_defaults(var=False)
parser.add_argument('--balance', dest='balance', action='store_true', help="balance samples in batch loading")
parser.set_defaults(balance=False)
parser.add_argument('--remote', dest='remote', action='store_true', help="use remote directories or not")
parser.set_defaults(remote=False)

args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


# Training
def train(epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader):
    net.train()
    net2.eval()  # fix one network and train the other

    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):
        try:
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        # labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1, 1), 1)
        # w_x = w_x.view(-1, 1).type(torch.FloatTensor)

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11 = net(inputs_u)
            outputs_u12 = net(inputs_u2)
            outputs_u21 = net2(inputs_u)
            outputs_u22 = net2(inputs_u2)

            # region classification sharpening
            # pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21,
            #                                                                                             dim=1) + torch.softmax(
            #     outputs_u22, dim=1)) / 4
            # ptu = pu ** (1 / args.T)  # temparature sharpening
            #
            # targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
            # endregion
            targets_u = (outputs_u11 + outputs_u12 + outputs_u21 + outputs_u22)/4
            targets_u = targets_u.detach()

            # label refinement of labeled samples
            outputs_x = net(inputs_x)
            outputs_x2 = net(inputs_x2)

            # region classification sharpening
            # px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            # px = w_x * labels_x + (1 - w_x) * px
            # ptx = px ** (1 / args.T)  # temparature sharpening
            #
            # targets_x = ptx / ptx.sum(dim=1, keepdim=True)  # normalize
            # endregion
            targets_x = (outputs_x + outputs_x2)/2
            targets_x = targets_x.detach()

        # mixmatch
        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1 - l)

        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        logits = net(mixed_input)
        logits_x = logits[:batch_size * 2]
        logits_u = logits[batch_size * 2:]

        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size * 2], logits_u, mixed_target[batch_size * 2:],
                                 epoch + batch_idx / num_iter, warm_up)

        # regularization
        # prior = torch.ones(args.num_class) / args.num_class
        # prior = prior.cuda()
        # pred_mean = torch.softmax(logits, dim=1).mean(0)
        # penalty = torch.sum(prior * torch.log(prior / pred_mean))

        loss = Lx + lamb * Lu #+ penalty
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'
                         % (args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx + 1, num_iter,
                            Lx.item(), Lu.item()))
        sys.stdout.flush()


def warmup(epoch, net, optimizer, dataloader):
    net.train()
    num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
    for batch_idx, (inputs, labels, _) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = MSEloss(outputs, labels)
        loss.backward()
        optimizer.step()

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t MSE-loss: %.4f'
                         % (args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx + 1, num_iter,
                            loss.item()))
        sys.stdout.flush()


def test(epoch, net1, net2):
    net1.eval()
    net2.eval()
    true = list()
    pred = list()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)
            outputs = outputs1 + outputs2
            predicted = torch.mean(outputs, 1)
            true.append(targets)
            pred.append(predicted)
    true = torch.vstack(true)
    pred = torch.vstack(pred)
    rmse = torch.sqrt(F.mse_loss(true, pred, reduction='none').mean(dim=1))
    pcc = 0 #TODO: fix pcc
    print("\n| Test Epoch #%d\t RMSE: %.2f%%\n" % (epoch, rmse))
    test_log.write('Epoch:%d   RMSE:%.2f\n' % (epoch, rmse))
    test_log.flush()


def eval_train(model, all_loss):
    model.eval()
    losses = torch.zeros(50000)
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = MSE(outputs, targets).mean(dim=1)
            for b in range(inputs.size(0)):
                losses[index[b]] = loss[b]
    losses = (losses - losses.min()) / (losses.max() - losses.min())
    all_loss.append(losses)

    if args.r == 0.9:  # average loss over last 5 epochs to improve convergence stability
        history = torch.stack(all_loss)
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1, 1)
    else:
        input_loss = losses.reshape(-1, 1)

    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss)
    prob = prob[:, gmm.means_.argmin()]
    return prob, all_loss


def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u * float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        Lx = torch.sqrt(torch.mean((outputs_u - targets_u) ** 2))
        Lu = torch.sqrt(torch.mean((outputs_x - targets_x) ** 2))

        return Lx, Lu, linear_rampup(epoch, warm_up)


def create_model():
    model = ResNet18(num_classes=2)
    model = model.cuda()
    return model


if __name__ == '__main__':
    stats_log = open('./checkpoint/%s_%.1f_%s' % (args.dataset, args.r, args.noise_mode) + '_stats.txt', 'w')
    test_log = open('./checkpoint/%s_%.1f_%s' % (args.dataset, args.r, args.noise_mode) + '_acc.txt', 'w')

    warm_up = 30

    print('| Building net')
    net1 = create_model()
    net2 = create_model()
    cudnn.benchmark = True

    criterion = SemiLoss()
    optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    MSE = nn.MSELoss(reduction='none')
    MSEloss = nn.MSELoss()

    all_loss = [[], []]  # save the history of losses from two networks

    for epoch in range(args.num_epochs + 1):
        lr = args.lr
        if epoch >= 150:
            lr /= 10
        for param_group in optimizer1.param_groups:
            param_group['lr'] = lr
        for param_group in optimizer2.param_groups:
            param_group['lr'] = lr
        loader = dataloader.AffectNetDataloader(
            batch_size=args.batch_size,
            num_workers=5,
            root_dir=args.data_path,
            log=stats_log
        )
        eval_loader = loader.run(mode='eval_train')
        test_loader = loader.run(mode='test')

        if epoch < warm_up:
            warmup_trainloader = loader.run('warmup')
            print('Warmup Net1')
            warmup(epoch, net1, optimizer1, warmup_trainloader)
            print('\nWarmup Net2')
            warmup(epoch, net2, optimizer2, warmup_trainloader)

        else:
            prob1, all_loss[0] = eval_train(net1, all_loss[0])
            prob2, all_loss[1] = eval_train(net2, all_loss[1])

            pred1 = (prob1 > args.p_threshold)
            pred2 = (prob2 > args.p_threshold)

            print('Train Net1')
            labeled_trainloader, unlabeled_trainloader = loader.run(mode='train', pred=pred2, prob=prob2)  # co-divide
            train(epoch, net1, net2, optimizer1, labeled_trainloader, unlabeled_trainloader)  # train net1

            print('\nTrain Net2')
            labeled_trainloader, unlabeled_trainloader = loader.run(mode='train', pred=pred1, prob=prob1)  # co-divide
            train(epoch, net2, net1, optimizer2, labeled_trainloader, unlabeled_trainloader)  # train net2

        test(epoch, net1, net2)