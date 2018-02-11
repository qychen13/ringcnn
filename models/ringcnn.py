import torch.nn as nn

from . import deeplab


class RingBlock(nn.Module):
    def __init__(self, base_module, rate):
        super(RingBlock, self).__init__()
        self.base_module = base_module
        self.rate = rate

    def forward(self, x):
        for i in range(self.rate):
            x = self.base_module(x)

        return x


def add_ring_blocks(model, rate):
    last_layer_name = None
    for name, _ in model.fulconv.named_models():
        last_layer_name = name

    last_layer = model.fulconv._modules[last_layer_name]
    model.fulconv._modules[last_layer_name] = RingBlock(last_layer, rate=rate)
