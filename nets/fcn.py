from torchvision import models
from torch import nn
import numpy as np
import torch


class VGG_FCN(nn.Module):
    def __init__(self, kwargs):
        super(VGG_FCN, self).__init__()
        out_c = kwargs['out_c']
        conv_sequential = list(models.vgg16(pretrained=kwargs['pretrain']).children())[0]
        modules_list = []
        for i in range(17):
            modules_list.append(conv_sequential._modules[str(i)])
        self.stage1 = nn.Sequential(*modules_list)

        modules_list = []
        for i in range(17, 24):
            modules_list.append(conv_sequential._modules[str(i)])
        self.stage2 = nn.Sequential(*modules_list)

        modules_list = []
        for i in range(24, 31):
            modules_list.append(conv_sequential._modules[str(i)])
        modules_list.append(nn.Conv2d(512, 4096, 1))
        modules_list.append(nn.Conv2d(4096, 4096, 1))
        self.stage3 = nn.Sequential(*modules_list)

        self.scores3 = nn.Conv2d(4096, out_c, 1)
        self.scores2 = nn.Conv2d(512, out_c, 1)
        self.scores1 = nn.Conv2d(256, out_c, 1)

        self.upsample_8x = nn.ConvTranspose2d(out_c, out_c, 16, 8, 4, bias=False)
        self.upsample_8x.weight.data = bilinear_kernel(out_c, out_c, 16)
        self.upsample_16x = nn.ConvTranspose2d(out_c, out_c, 4, 2, 1, bias=False)
        self.upsample_16x.weight.data = bilinear_kernel(out_c, out_c, 4)
        self.upsample_32x = nn.ConvTranspose2d(out_c, out_c, 4, 2, 1, bias=False)
        self.upsample_32x.weight.data = bilinear_kernel(out_c, out_c, 4)

    def forward(self, x):
        x = self.stage1(x)
        s1 = x

        x = self.stage2(x)
        s2 = x

        x = self.stage3(x)
        s3 = x

        s3 = self.scores3(s3)
        s3 = self.upsample_32x(s3)

        s2 = self.scores2(s2)
        s2 = s2 + s3
        s2 = self.upsample_16x(s2)

        s1 = self.scores1(s1)
        s = s1 + s2
        s = self.upsample_8x(s)

        return s



class VGG_BN_FCN(nn.Module):
    def __init__(self, kwargs):
        super(VGG_BN_FCN, self).__init__()
        out_c = kwargs['out_c']
        conv_sequential = list(models.vgg16_bn(pretrained=kwargs['pretrain']).children())[0]
        modules_list = []
        for i in range(24):
            modules_list.append(conv_sequential._modules[str(i)])
        self.stage1 = nn.Sequential(*modules_list)

        modules_list = []
        for i in range(24, 34):
            modules_list.append(conv_sequential._modules[str(i)])
        self.stage2 = nn.Sequential(*modules_list)

        modules_list = []
        for i in range(34, 44):
            modules_list.append(conv_sequential._modules[str(i)])
        modules_list.append(nn.Conv2d(512, 4096, 1))
        modules_list.append(nn.Conv2d(4096, 4096, 1))
        self.stage3 = nn.Sequential(*modules_list)

        self.scores3 = nn.Conv2d(4096, out_c, 1)
        self.scores2 = nn.Conv2d(512, out_c, 1)
        self.scores1 = nn.Conv2d(256, out_c, 1)

        self.upsample_8x = nn.ConvTranspose2d(out_c, out_c, 16, 8, 4, bias=False)
        self.upsample_8x.weight.data = bilinear_kernel(out_c, out_c, 16)
        self.upsample_16x = nn.ConvTranspose2d(out_c, out_c, 4, 2, 1, bias=False)
        self.upsample_16x.weight.data = bilinear_kernel(out_c, out_c, 4)
        self.upsample_32x = nn.ConvTranspose2d(out_c, out_c, 4, 2, 1, bias=False)
        self.upsample_32x.weight.data = bilinear_kernel(out_c, out_c, 4)

    def forward(self, x):
        x = self.stage1(x)
        s1 = x

        x = self.stage2(x)
        s2 = x

        x = self.stage3(x)
        s3 = x

        s3 = self.scores3(s3)
        s3 = self.upsample_32x(s3)

        s2 = self.scores2(s2)
        s2 = s2 + s3
        s2 = self.upsample_16x(s2)

        s1 = self.scores1(s1)
        s = s1 + s2
        s = self.upsample_8x(s)

        return s



class Res_FCN(nn.Module):
    def __init__(self, kwargs):
        super(Res_FCN, self).__init__()
        out_c = kwargs['out_c']
        base_model = list(models.resnet34(pretrained=kwargs['pretrain']).children())
        self.stage1 = nn.Sequential(*base_model[:-4])
        self.stage2 = base_model[-4]
        self.stage3 = base_model[-3]

        self.scores1 = nn.Conv2d(512, out_c, 1)
        self.scores2 = nn.Conv2d(256, out_c, 1)
        self.scores3 = nn.Conv2d(128, out_c, 1)

        self.upsample_8x = nn.ConvTranspose2d(out_c, out_c, 16, 8, 4, bias=False)
        self.upsample_8x.weight.data = bilinear_kernel(out_c, out_c, 16)  # 使用双线性 kernel

        self.upsample_4x = nn.ConvTranspose2d(out_c, out_c, 4, 2, 1, bias=False)
        self.upsample_4x.weight.data = bilinear_kernel(out_c, out_c, 4)  # 使用双线性 kernel

        self.upsample_2x = nn.ConvTranspose2d(out_c, out_c, 4, 2, 1, bias=False)
        self.upsample_2x.weight.data = bilinear_kernel(out_c, out_c, 4)  # 使用双线性 kernel

    def forward(self, x):
        x = self.stage1(x)
        s1 = x

        x = self.stage2(x)
        s2 = x

        x = self.stage3(x)
        s3 = x

        s3 = self.scores1(s3)
        s3 = self.upsample_2x(s3)
        s2 = self.scores2(s2)
        s2 = s2 + s3

        s1 = self.scores3(s1)
        s2 = self.upsample_4x(s2)

        s = self.upsample_8x(s1 + s2)
        return s


def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)
