import sys
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import random
import argparse
import numpy as np
# from PreResNet import *
# from InceptionResNetV2 import *
from ResNet18 import ResNet18
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
import dataloader_AffectNet as dataloader
import utils

parser = argparse.ArgumentParser(description='PyTorch AffectNet Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize')
parser.add_argument('--num_class', default=2, type=int)
parser.add_argument('--data_path', default='', type=str, help='path to dataset')
parser.add_argument('--dataset', default='affectnet', type=str)
parser.add_argument('--regression', dest='regression', action='store_true',
                    help="train on regression task, using the dimensional A/V model")
parser.set_defaults(regression=False)
parser.set_defaults(var=False)
parser.add_argument('--multigpu', dest='multigpu', action='store_true', help='use nn.DataParallel')
parser.set_defaults(multigpu=False)
parser.add_argument('--savedata', dest='savedata', action='store_true', help='use nn.DataParallel')
parser.set_defaults(savedata=False)

parser.add_argument('--model_path', type=str)
parser.add_argument('--partition', type=str, default=None)

parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)

args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


# def test(net):
#     net.eval()
#     true = list()
#     pred = list()
#     with torch.no_grad():
#         for batch_idx, (inputs, targets, _) in enumerate(test_loader):
#             inputs, targets = inputs.cuda(), targets.cuda()
#             outputs1 = net(inputs)
#             true.append(targets)
#             pred.append(outputs1)
#     true = torch.vstack(true)
#     pred = torch.vstack(pred)
#     if args.savedata:
#         np.savez('checkpoint/data.npz', pred.cpu(), true.cpu())
#     rmse = torch.sqrt(F.mse_loss(true, pred, reduction='none').mean(dim=1))
#     pcc = utils.PCC(true, pred)
#     ccc = utils.CCC(true, pred)
#     print(
#         "\n| Test \t Arr RMSE: {}, Val RMSE:  {}\n Arr PCC: {} Val PCC: {} \n Arr CCC: {} Val CCC: {} \n ".format(
#             rmse[0],
#             rmse[1],
#             pcc[0],
#             pcc[1],
#             ccc[0],
#             ccc[1]
#         )
#     )

def get_hist(model):
    model.eval()
    clean_size, noisy_size = len(clean_loader.dataset), len(noisy_loader.dataset)
    clean_losses = torch.zeros(clean_size)
    noisy_losses = torch.zeros(noisy_size)
    with torch.no_grad():

        num_iter = clean_size // clean_loader.batch_size + 1
        for batch_idx, (inputs, targets, index) in enumerate(clean_loader):
            # exp = targets[:, 2].cuda().long()
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = PSLoss(outputs, targets)
            for b in range(inputs.size(0)):
                clean_losses[index[b]] = loss[b]
            sys.stdout.write('\r')
            sys.stdout.write('%s: Clean Data | Iter[%3d/%3d]' % (args.dataset, batch_idx + 1, num_iter))
            sys.stdout.flush()

        num_iter = noisy_size // noisy_loader.batch_size + 1
        for batch_idx, (inputs, targets, index) in enumerate(noisy_loader):
            # exp = targets[:, 2].cuda().long()
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = PSLoss(outputs, targets)
            for b in range(inputs.size(0)):
                noisy_losses[index[b]] = loss[b]
            sys.stdout.write('\r')
            sys.stdout.write('%s: Noisy Data | Iter[%3d/%3d]' % (args.dataset, batch_idx + 1, num_iter))
            sys.stdout.flush()

    losses = torch.cat([clean_losses, noisy_losses])
    losses = (losses - losses.min()) / (losses.max() - losses.min())

    losses = losses.reshape(-1, 1)
    losses = losses.cpu().numpy()
    gmm = GaussianMixture(n_components=2, max_iter=15, reg_covar=5e-4, tol=1e-2)
    gmm.fit(losses)

    plt.hist(losses[:clean_size], bins=20, alpha=0.7, label='clean', density=False)
    plt.hist(losses[clean_size:], bins=20, alpha=0.6, label='noisy', density=False)
    x = np.linspace(0, 1, 1000).reshape(-1, 1)
    logprob = gmm.score_samples(x)
    pdf = np.exp(logprob)
    plt.twinx()
    plt.plot(x, pdf, '-k', label='GMM')
    plt.legend()
    plt.savefig('histograms.png')


def create_model(model_pth: str):
    model = ResNet18(do_regr=True, do_cls=False, variance=False, pretrained=True)
    state_dct = torch.load(model_pth, map_location=torch.device('cpu'))
    new_state = dict()
    for key in state_dct:
        new_state[key.replace('module.', '')] = state_dct[key]
    model.load_state_dict(new_state)
    if args.multigpu:
        # torch.cuda.set_per_process_memory_fraction(0.4, device=0)
        model = nn.DataParallel(model)
    model = model.cuda()
    return model


if __name__ == '__main__':
    net1 = create_model(args.model_path)
    cudnn.benchmark = True
    PSLoss = nn.L1Loss(reduction='none')

    clean_data = dataloader.AffectNetDataloader(
        batch_size=args.batch_size,
        num_workers=5,
        root_dir=args.data_path,
        log=None,
        artifitial_noise='clean'
    )
    noisy_data = dataloader.AffectNetDataloader(
        batch_size=args.batch_size,
        num_workers=5,
        root_dir=args.data_path,
        log=None,
        artifitial_noise='noisy'
    )
    clean_loader = clean_data.run(mode='eval_train')
    noisy_loader = noisy_data.run(mode='eval_train')
    get_hist(net1)
