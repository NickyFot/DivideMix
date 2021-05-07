import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet


class ResNet18(nn.Module):
    def __init__(self, do_regr, do_cls, variance=False, pretrained=True):
        super().__init__()
        self.do_regr = do_regr
        self.do_cls = do_cls
        self.do_var = variance
        self.num_classes_regr = 2
        self.num_classes_cls = 8

        backbone = resnet.resnet18(pretrained=pretrained)
        self.backbone = torch.nn.Sequential(*(list(backbone.children())[:-1]))
        self.linear_regr = nn.Linear(512, 2)
        if self.do_var:
            self.linear_var = nn.Linear(512, 2)
            def mini_init(m):
                if hasattr(m, 'weight'):
                    nn.init.normal_(m.weight, mean=0, std=0.0001)

            self.linear_var.apply(mini_init)

    def forward(self, x):
        out = self.backbone(x)
        out = out.view(out.size(0), -1)
        preds = {}
        preds['feats'] = out
        if self.do_regr:
            out_regr = self.linear_regr(out)
            preds['regr'] = out_regr
        if self.do_cls:
            out_cls = self.linear_cls(out)
            preds['cls'] = out_cls

        if self.do_var:
            var = self.linear_var(out)
            preds['var'] = var
        return preds['regr']  # TODO: handle dict for variance
