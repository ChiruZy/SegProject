from torch import nn
from torch import sigmoid


class DiceCoeff(nn.Module):
    def __init__(self):
        super(DiceCoeff, self).__init__()

    def forward(self, input, target):
        batch_size = target.size(0)
        smooth = 1

        input_flat = sigmoid(input.view(batch_size, -1))
        target_flat = target.view(batch_size, -1)

        intersection = input_flat * target_flat

        coeff = 2 * (intersection.sum(1) + 1e-5) / (input_flat.sum(1) + target_flat.sum(1) + 1e-5)
        return coeff.sum() / batch_size


class MultiDiceCoeff(nn.Module):
    def __init__(self):
        super(MultiDiceCoeff, self).__init__()

    def forward(self, input, target):
        C = target.shape[1]
        dice = DiceCoeff()
        coeff = [dice(input[:, i], target[:, i]) for i in range(C)]
        return coeff


def dice_coeff(input, target):
    return DiceCoeff()(input, target) if target.shape[1] == 1 else MultiDiceCoeff()(input, target)


def dice_loss(input, target, weight=None):
    if target.shape[1] == 1:
        return 1 - DiceCoeff()(input, target)
    else:
        total = 0
        for idx, loss in enumerate(MultiDiceCoeff()(input, target)):
            total += 1 - loss if not weight else (1 - loss) * weight[idx]
        return total


def miou(pred, target, n_classes=1):
    ious = []
    for cls in range(1, n_classes+1):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum().item()
        union = pred_inds.sum().item() + target_inds.sum().item() - intersection
        ious.append(0 if union == 0 else intersection / union)
    return ious



if __name__ == '__main__':
    import torch
    pre = torch.tensor([[[[-10, -10], [1, -3.3]]]])
    mask = torch.tensor([[[[1, 1], [0, 1]]]])

    print(pre, mask)
    print(dice_loss(pre, mask))

