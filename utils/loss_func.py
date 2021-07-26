import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def make_one_hot(target, n_cls):
    shape = np.array(target.shape)
    shape[1] = n_cls
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, target.cpu(), 1)

    return result


class BinaryDiceLoss(nn.Module):
    def __init__(self, need_sigmoid=True, eps=1e-8):
        super(BinaryDiceLoss, self).__init__()
        self.eps = eps
        self.need_sigmoid = need_sigmoid

    def forward(self, predict, target):
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        if self.need_sigmoid:
            predict = torch.sigmoid(predict)
        num = torch.sum(torch.mul(predict, target), dim=1)
        den = torch.sum(predict + target, dim=1) + self.eps

        loss = 1 - num / den
        return loss.sum()


class DiceLoss(nn.Module):
    def __init__(self, weight=None, ignore_idx=None):
        super(DiceLoss, self).__init__()
        self.weight = weight
        self.ignore_idx = ignore_idx

    def forward(self, predict, target):
        dice = BinaryDiceLoss(need_sigmoid=False)
        total_loss = 0

        predict = F.softmax(predict, dim=1)
        target = torch.from_numpy(make_one_hot(target, 1))

        for i in range(target.shape[1]):
            if i in self.ignore_idx:
                continue

            dice_loss = dice(predict[:, i], target[:, i])
            if self.weight is not None:
                dice_loss *= self.weights[i]
            total_loss += dice_loss
        return total_loss/target.shape[1]


def dice_loss(pred, target, n_cls=1, weight=None, ignore_idx=None):
    if n_cls == 1:
        return BinaryDiceLoss()(pred, target)
    else:
        return DiceLoss(weight, ignore_idx)


def miou(pred, target, n_classes=1):
    ious = 0
    for cls in range(1, n_classes+1):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum().item()
        union = pred_inds.sum().item() + target_inds.sum().item() - intersection
        ious += 0 if union == 0 else intersection / union
    return ious / n_classes
