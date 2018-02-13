import torch.nn as nn


class RingBlock(nn.Module):
    def __init__(self, base_module, rate, residual=False):
        super(RingBlock, self).__init__()
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


def add_ring_blocks(model, layer_names, rates):
    for layer_name, rate in zip(layer_names, rates):
        model._modules[layer_name] = RingBlock(model._modules[layer_name], rate=rate)


def delete_dilated_conv(model):
    for n, m in model.named_modules():
        if 'conv2' in n:
            if m.dilation != 1:
                m.dilation = (1, 1)
                m.padding = (1, 1)


def ringcnn_deeplab(model, base_rate=2):
    l = len(model.fulconv._modules)
    rates = [base_rate * (2 ** i) for i in range(l)]
    layer_names = [str(i) for i in range(l)]
    add_ring_blocks(model.fulconv, layer_names, rates)
    return model


def ringcnn_dilatedfcn(model, base_rate=2):
    add_ring_blocks(model, ('layer3', 'layer4'), (base_rate, base_rate * 2))
    return model
