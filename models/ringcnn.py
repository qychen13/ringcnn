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
                x = self.base_module(x)
        return x


def add_ring_blocks(model, layer_names, rate):
    for layer_name in layer_names:
        model._modules[layer_name] = RingBlock(model._modules[layer_name], rate=rate)


def ringcnn_deeplab(model, rate):
    l = len(model.fulconv._modules)
    add_ring_blocks(model.fulconv, (l - 1,), rate)
    return model


def ringcnn_dilatedfcn(model, rate):
    add_ring_blocks(model, ('layer4',), rate)
    return model
