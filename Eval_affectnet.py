from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler
from torch.distributions.normal import Normal
import random
import os
import argparse
import numpy as np
# from PreResNet import *
# from InceptionResNetV2 import *
from ResNet18 import ResNet18
from sklearn.mixture import GaussianMixture

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
parser.add_argument('--model_path', type=str)
parser.add_argument('--partition', type=str, default=None)

parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)

args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


def test(net):
    net.eval()
    true = list()
    pred = list()
    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net(inputs)
            true.append(targets)
            pred.append(outputs1)
    true = torch.vstack(true)
    pred = torch.vstack(pred)
    rmse = torch.sqrt(F.mse_loss(true, pred, reduction='none').mean(dim=1))
    pcc = utils.PCC(true, pred)
    ccc = utils.CCC(true, pred)
    print(
        "\n| Test \t Arr RMSE: {}, Val RMSE:  {}\n Arr PCC: {} Val PCC: {} \n Arr CCC: {} Val CCC: {} \n ".format(
            rmse[0],
            rmse[1],
            pcc[0],
            pcc[1],
            ccc[0],
            ccc[1]
        )
    )


def create_model(model_pth: str):
    model = ResNet18(True, False, variance=False, pretrained=False)
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
    MSEloss = nn.MSELoss()

    all_loss = [[], []]  # save the history of losses from two networks

    loader = dataloader.AffectNetDataloader(
        batch_size=args.batch_size,
        num_workers=5,
        root_dir=args.data_path,
        log=None,
        partition=args.partition
    )
    test_loader = loader.run(mode='test')
    test(net1)
