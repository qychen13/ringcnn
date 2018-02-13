"""
copy from https://github.com/ZijunDeng/pytorch-semantic-segmentation/blob/master/models/psp_net.py and modified
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models.resnet as resnet


class DilatedFCN(resnet.ResNet):
    def __init__(self, num_classes, layers):
        super(DilatedFCN, self).__init__(resnet.Bottleneck, layers, num_classes)
        del self.avgpool, self.fc

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        self.fulconv = None
        self.logits = nn.Sequential(nn.Dropout2d(0.5),
                                    nn.Conv2d(2048, num_classes, kernel_size=1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.logits(x)
        return x


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_dim, setting):
        super(PyramidPoolingModule, self).__init__()
        self.features = []
        reduction_dim = in_dim // len(setting)
        for s in setting:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim, momentum=.95),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.upsample(f(x), x_size[2:], mode='bilinear'))
        out = torch.cat(out, 1)
        return out


class PSPNet(DilatedFCN):
    def __init__(self, num_classes, layers, psp_sizes=(1, 2, 3, 6), use_aux=True):
        super(PSPNet, self).__init__(num_classes, layers)
        fulconv = [PyramidPoolingModule(2048, psp_sizes),
                   nn.Conv2d(4096, 512, kernel_size=3, padding=1, bias=False),
                   nn.BatchNorm2d(512, momentum=.95),
                   nn.ReLU(inplace=True)]
        self.fulconv = nn.Sequential(*fulconv)
        self.use_aux = use_aux
        if use_aux:
            self.aux_logits = nn.Sequential(nn.Dropout2d(0.5),
                                        nn.Conv2d(1024, num_classes, kernel_size=1))
        self.logits = nn.Sequential(nn.Dropout2d(0.5),
                                    nn.Conv2d(512, num_classes, kernel_size=1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.training and self.use_aux:
            aux = self.aux_logits(x)
        x = self.layer4(x)
        x = self.fulconv(x)
        x = self.logits(x)
        if self.training and self.use_aux:
            return x, aux
        return x


layers = {'resnet50': [3, 4, 6, 3], 'resnet101': [3, 4, 23, 3]}


def pspnet(model_name, num_classes):
    return PSPNet(num_classes, layers[model_name], use_aux=False)


def dilatedFCN(model_name, num_classes):
    return DilatedFCN(num_classes, layers[model_name])
