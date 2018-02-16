import torch.nn as nn

import copy

import utilities.initialization


class RingModule(nn.Module):
    def __init__(self, base_module, rate, residual=False):
        super(RingModule, self).__init__()
        self.base_module = base_module
        self.rate = rate
        self.residual = residual

    def forward(self, x):
        for i in range(self.rate):
            if self.residual:
                x = 0.5 * x + 0.5 * self.base_module(x)
            else:
                x = self.base_module(x) * 0.5
        return x

class RingBottleNeck(nn.Module):
    expansion = 4

    def __init__(self, residual_bottleneck, rate, residual=False):
        super(RingBottleNeck, self).__init__()
        self.rate = rate
        self.residual = residual
        self.upsample = None
        for n, m in residual_bottleneck.named_children():
            if 'downsample' in n:
                if m.stride != 1:
                    self.upsample = nn.Sequential(nn.Conv2d(m.outchanels, m.inchanels, 1),
                                                  nn.Upsample(scale_factor=m.stride, mode='bilinear'))
                else:
                    self.upsample = nn.Conv2d(m.outchanels, m.inchanels, 1)
                utilities.initialization.init_network(self.upsample)

            else:
                self.add_module(n, copy.deepcopy(m))

    def base_forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.relu(out)
        if self.upsample is not None:
            out = self.upsample(out)

        return out

    def forward(self, x):
        for i in range(self.rate):
            if self.residual:
                x = 0.5 * self.base_forward(x) + 0.5 * x
            else:
                x = self.base_forward(x)

        return x


def add_ring_blocks_v1(model, layer_names, rates):
    for layer_name, rate in zip(layer_names, rates):
        model._modules[layer_name] = RingModule(model._modules[layer_name], rate=rate)


def add_ring_blocks_v2(model, layer_names, rates):
    for layer_name, rate in zip(layer_names, rates):
        for n, m in model._modules[layer_name].named_children():
            model._modules[n] = RingBottleNeck(m, rate=rate)


def delete_dilated_conv(model):
    for n, m in model.named_modules():
        if 'conv2' in n:
            if m.dilation != 1:
                m.dilation = (1, 1)
                m.padding = (1, 1)


def ringcnn_v1_deeplab(model, base_rate=2):
    l = len(model.fulconv._modules)
    rates = [base_rate * (2 ** i) for i in range(l)]
    layer_names = [str(i) for i in range(l)]
    add_ring_blocks_v1(model.fulconv, layer_names, rates)
    return model


def ringcnn_v2_deeplab(model, base_rate=2):
    l = len(model.fulconv._modules)
    rates = [base_rate * (2 ** i) for i in range(l)]
    layer_names = [str(i) for i in range(l)]
    add_ring_blocks_v2(model.fulconv, layer_names, rates)
    return model


def ringcnn_v1_dilatedfcn(model, base_rate=2):
    add_ring_blocks_v1(model, ('layer3', 'layer4'), (base_rate, base_rate * 2))
    return model
