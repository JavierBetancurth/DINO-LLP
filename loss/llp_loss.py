import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.constraints import simplex

# from llp_vat.lib.networks import GaussianNoise


def compute_soft_kl(inputs, targets):
    with torch.no_grad():
        loss = cross_entropy_loss(inputs, targets)
        loss = torch.sum(loss, dim=-1).mean()
    return loss


def compute_hard_l1(inputs, targets, num_classes):
    with torch.no_grad():
        predicted = torch.bincount(inputs.argmax(1),
                                   minlength=num_classes).float()
        predicted = predicted / torch.sum(predicted, dim=0)
        targets = torch.mean(targets, dim=0)
        loss = F.l1_loss(predicted, targets, reduction="sum")
    return loss


def cross_entropy_loss(input, target, eps=1e-8):
    assert simplex.check(input) and simplex.check(target), \
        "input {} and target {} should be a simplex".format(input, target)
    input = torch.clamp(input, eps, 1 - eps)
    loss = -target * torch.log(input)
    return loss


class ProportionLoss(nn.Module):
    def __init__(self, metric, alpha, eps=1e-8):
        super(ProportionLoss, self).__init__()
        self.metric = metric
        self.eps = eps
        self.alpha = alpha

    def forward(self, input, target):
        # input and target shoud ba a probability tensor
        # and have been averaged over bag size
        assert simplex.check(input) and simplex.check(target), \
            "input {} and target {} should be a simplex".format(input, target)
        assert input.shape == target.shape

        if self.metric == "ce":
            loss = cross_entropy_loss(input, target, eps=self.eps)
        elif self.metric == "l1":
            loss = F.l1_loss(input, target, reduction="none")
        elif self.metric == "mse":
            loss = F.mse_loss(input, target, reduction="none")
        else:
            raise NameError("metric {} is not supported".format(self.metric))

        loss = torch.sum(loss, dim=-1).mean()
        return self.alpha * loss
