import torchnet.engine as engine
import torch
import torch.backends.cudnn
import torch.nn as nn

from tqdm import tqdm


class Engine(engine.Engine):
    # minor change
    def __init__(self, gpu_ids, network, criterion, train_iterator, validate_iterator, optimizer):
        self.state = {
            'gpu_ids': gpu_ids,
            'network': network,
            'train_iterator': train_iterator,
            'validate_iterator': validate_iterator,
            'maxepoch': None,
            'criterion': criterion,
            'optimizer': optimizer,
            'epoch': None,
            't': None,
            'train': True,
            'output': None,
            'loss': None
        }

        # set cudnn bentchmark
        torch.backends.cudnn.benchmark = True

        super(Engine, self).__init__()

    def forward(self):
        state = self.state
        ipt, target = state['sample'][0], state['sample'][1]
        self.hook('on_start_forward', state)
        if state['gpu_ids'] is not None:
            output = nn.parallel.data_parallel(state['network'], ipt, state['gpu_ids'])
        else:
            output = state['network'](ipt)
        loss = state['criterion'](output, target)
        state['output'] = output
        state['loss'] = loss
        self.hook('on_end_forward', state)

        # to free memory in save_for_backward
        state['output'] = None
        state['loss'] = None

        return loss

    def resume(self, maxepoch, epoch, t):
        state = self.state

        state['epoch'] = epoch
        state['t'] = t
        state['maxepoch'] = maxepoch
        state['train'] = True

        self.hook('on_start', state)

        while state['epoch'] < state['maxepoch']:
            self.hook('on_start_epoch', state)

            for sample in tqdm(state['train_iterator']):
                state['sample'] = sample
                self.hook('on_end_sample', state)

                loss = self.forward()

                state['optimizer'].zero_grad()
                loss.backward()
                state['optimizer'].step()

                self.hook('on_end_update', state)
                state['t'] += 1
            state['epoch'] += 1
            self.hook('on_end_epoch', state)
        self.hook('on_end', state)
        return state

    def train(self, maxepoch):
        self.resume(maxepoch, 0, 0)

    def validate(self):
        state = self.state

        state['train'] = False
        self.hook('on_start', state)

        for sample in tqdm(state['validate_iterator']):
            state['sample'] = sample
            self.hook('on_end_sample', state)
            self.forward()

        self.hook('on_end_test', state)
        self.hook('on_end', state)
        return state
