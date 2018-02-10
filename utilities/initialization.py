import torch.nn as nn
from  torch.nn.parameter import Parameter
import math


def init_network(model):
    print('==> Network initialization.')
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.reset_parameters()
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            n = m.weight.size(1)
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


def load_partial_network(model, state_dict):
    """Copies from load_state_dict
    """
    print('==> Load Partial Network...')
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        try:
            own_state[name].copy_(param)
        except:
            print('While copying the parameter named {}, whose dimensions in the model are'
                  ' {} and whose dimensions in the checkpoint are {}, ...'.format(
                name, own_state[name].size(), param.size()))
            raise

    missing = set(own_state.keys()) - set(state_dict.keys())
    more = set(state_dict.keys()) - set(own_state.keys())
    print('******Not init {}******'.format(missing))
    print('******Not load {}******'.format(more))
