from __future__ import print_function

import os
import sys
from datetime import datetime
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
from torch.distributions.normal import Normal
import random
import argparse
import numpy as np
from ResNet18 import ResNet18
from sklearn.mixture import GaussianMixture

import dataloader_AffectNet as dataloader
import utils

parser = argparse.ArgumentParser(description='PyTorch AffectNet Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize')
parser.add_argument('--lr', '--learning_rate', default=0.0001, type=float, help='initial learning rate')
parser.add_argument('--noise_mode', default='sym')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=50, type=int)
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
parser.add_argument('--multigpu', dest='multigpu', action='store_true', help='use nn.DataParallel')
parser.set_defaults(multigpu=False)

args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


# Training
def train(epoch, net, optimizer, labeled_trainloader, unlabeled_trainloader):
    net.train()

    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1
    for batch_idx, (inputs_x, inputs_x2, labels_x, w_x) in enumerate(labeled_trainloader):
        optimizer.zero_grad()
        try:
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            inputs_u, inputs_u2 = unlabeled_train_iter.next()
        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        # labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1, 1), 1)
        w_x = w_x.view(-1, 1).type(torch.FloatTensor)
        inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            # label guessing of unlabeled samples and augmentations
            outputs_u11 = net(inputs_u)
            outputs_u12 = net(inputs_u2)
            targets_u = (outputs_u11 + outputs_u12)/2
            targets_u = targets_u.detach()

            targets_x = labels_x.detach()

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
        with torch.cuda.amp.autocast():
            logits = net(mixed_input)
            logits_x = logits[:batch_size * 2]
            logits_u = logits[batch_size * 2:]

            Lx, Lu, lamb = criterion(
                logits_x,
                mixed_target[:batch_size * 2],
                logits_u,
                mixed_target[batch_size * 2:],
                epoch + batch_idx / num_iter,
                warm_up
            )
            loss = Lx + lamb * Lu
            # compute gradient and do SGD step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        sys.stdout.write('\r')
        sys.stdout.write(
            '{}:{}-{} | Epoch [{}/{}] Iter[{}/{}]\t Labeled loss: {}  Unlabeled loss: {} '.format(
                args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx + 1, num_iter, Lx.mean().item(), Lu.mean().item()
            )
        )
        sys.stdout.flush()


def warmup(epoch, net, optimizer, dataloader):
    net.train()
    num_iter = (len(dataloader.dataset) // dataloader.batch_size) + 1
    for batch_idx, (inputs, labels, _) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = net(inputs)
            loss = MSEloss(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t MSE-loss: %.4f'
                         % (args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx + 1, num_iter,
                            loss.item()))
        sys.stdout.flush()


def test(epoch, net1):
    net1.eval()
    true = list()
    pred = list()
    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            true.append(targets)
            pred.append(outputs1)
    true = torch.vstack(true)
    pred = torch.vstack(pred)
    rmse = torch.sqrt(F.mse_loss(true, pred, reduction='none').mean(dim=1))
    pcc = utils.PCC(true, pred)
    print("\n| Test Epoch #{}\t Arr RMSE: {}, Val RMSE:  {}\n Arr PCC: {} Val PCC: {} \n".format(epoch, rmse[0], rmse[1], pcc[0], pcc[1]))
    test_log.write('Epoch:{}   Arr RMSE: {}, Val RMSE:  {}, Arr PCC: {} Val PCC: {} \n'.format(epoch, rmse[0], rmse[1], pcc[0], pcc[1]))
    test_log.flush()


def eval_train(model, all_loss) -> (list, list):
    model.eval()
    samples_size = len(eval_loader.dataset)
    losses = torch.zeros(samples_size)
    pred, true, expression = list(), list(), list()
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            exp = targets[:, 2]
            inputs, targets = inputs.cuda(), targets[:, :2].cuda()
            outputs = model(inputs)
            pred.append(outputs)
            true.append(targets)
            expression.append(exp)
            loss = MSE(outputs, targets).mean(dim=1)
            for b in range(inputs.size(0)):
                losses[index[b]] = loss[b]
    pred = torch.vstack(pred)
    true = torch.vstack(true)
    expression = torch.cat(expression)
    np.savez('checkpoint/data.npz', pred.cpu(), true.cpu(), expression)
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
        Lu = F.mse_loss(outputs_u, targets_u)
        Lx = F.mse_loss(outputs_x, targets_x)

        return Lx, Lu, linear_rampup(epoch, warm_up)


def create_model():
    model = ResNet18(True, False, variance=False, pretrained=True)
    if args.multigpu:
        # torch.cuda.set_per_process_memory_fraction(0.4, device=0)
        model = nn.DataParallel(model)
    model = model.cuda()
    return model


def save_model(epoch, model, model_num):
    torch.save(model.state_dict(), log_folder +'%s_lr%.1f_epoch%s_ensemble%s' % (args.dataset, args.r, str(epoch), str(model_num)) + '_model.pth')


def calculate_prior():
    label_dist = Normal(torch.tensor([0.1123, 0.1980]), torch.tensor([0.2989, 0.5137]))  # mean and std of arousal and valence in affectnet
    dx = torch.arange(-1, 1, 0.01)
    dx = torch.vstack([dx, dx]).permute(1, 0)
    p = label_dist.log_prob(dx)
    return p.cuda(), dx


if __name__ == '__main__':
    start_time = datetime.now().strftime('%Y%m%d_%H%M')
    log_folder = './checkpoint/' + start_time + '/'
    os.mkdir(log_folder)
    test_log = open(log_folder + '%s_%.1f_%s' % (args.dataset, args.r, args.noise_mode) + '_acc.txt', 'w')

    warm_up = 30

    scaler = GradScaler()
    print('| Building net')
    net = create_model()
    cudnn.benchmark = True

    prior, dx = calculate_prior()
    criterion = SemiLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=5e-4)

    MSE = nn.MSELoss(reduction='none')
    MSEloss = nn.MSELoss()

    all_loss = [[], []]  # save the history of losses from two networks

    for epoch in range(args.num_epochs + 1):
        lr = args.lr
        if epoch >= 100:
            lr /= 10
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        loader = dataloader.AffectNetDataloader(
            batch_size=args.batch_size,
            num_workers=5,
            root_dir=args.data_path,
            log=None
        )
        eval_loader = loader.run(mode='eval_train')
        test_loader = loader.run(mode='test')

        if epoch < warm_up:
            warmup_trainloader = loader.run('warmup')
            print('Warmup Net1')
            warmup(epoch, net, optimizer, warmup_trainloader)
        else:
            prob, all_loss[0] = eval_train(net, all_loss[0])

            pred: list = (prob > args.p_threshold)

            print('Train Net')
            labeled_trainloader, unlabeled_trainloader = loader.run(mode='train', pred=pred, prob=prob)
            train(epoch, net, optimizer, labeled_trainloader, unlabeled_trainloader)
        test(epoch, net)
        save_model(epoch, net, 0)

# python Train_affectnet.py --batch_size 32 --multigpu --data_path /import/nobackup_mmv_ioannisp/shared/datasets/AffectNet/ --lambda_u 1 --alpha 1 --r 0.9
