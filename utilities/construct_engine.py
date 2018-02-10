""" a lot copy from tnt.example
"""
import os
import time
import copy

import torch
from torch.autograd.variable import Variable

from torchnet.logger import VisdomPlotLogger
import torchnet.meter as meter

from .engine import Engine


def construct_engine(engine_args, checkpoint_iter_freq=5000, checkpoint_epoch_freq=1,
                     checkpoint_save_path='checkpoints',
                     iter_log_freq=30, environment='main', lr_points=[]):
    engine = Engine(**engine_args)

    # ***************************Meter Setting******************************

    class Meterhelper(object):
        # titles: dict for {key: title} pair

        def __init__(self, meter, titles, plot_type='line'):
            self.meter = meter

            assert type(titles) is dict
            self.loggers = dict()
            for key in titles:
                self.loggers[key] = VisdomPlotLogger(plot_type, opts={'title': titles[key]}, env=environment)

        def log(self, key, x, y_arg=None):
            assert key in self.loggers.keys()
            if y_arg is None:
                y = self.meter.value()
            else:
                y = self.meter.value(y_arg)
            if type(y) is tuple:
                y = y[0]
            self.loggers[key].log(x, y)

        def add(self, *arg, **args):
            return self.meter.add(*arg, **args)

        def reset(self):
            return self.meter.reset()

    time_meter = meter.TimeMeter(1)

    meters = dict(
        data_loading_meter=Meterhelper(meter.MovingAverageValueMeter(windowsize=10), dict(data_t='Data Loading Time')),
        gpu_time_meter=Meterhelper(meter.MovingAverageValueMeter(windowsize=10), dict(gpu_t='Gpu Computing Time')),
        train_loss_meter=Meterhelper(meter.MovingAverageValueMeter(windowsize=10),
                                     dict(train_loss_iteration='Training Loss(Iteration)',
                                          train_loss_epoch='Training Loss(Epoch)')),
        test_loss_meter=Meterhelper(meter.AverageValueMeter(), dict(test_loss='Test Loss')))

    # ***************************Auxiliaries******************************

    def reset_meters():
        time_meter.reset()
        for key in meters:
            meters[key].reset()

    def prepare_network(state):
        # switch model
        if state['train']:
            state['network'].train()
        else:
            state['network'].eval()

    def wrap_data(state):
        if state['gpu_ids'] is not None:
            # state['sample'][0] = state['sample'][0].cuda(device=state['gpu_ids'][0], async=False)
            state['sample'][1] = state['sample'][1].cuda(device=state['gpu_ids'][0], async=True)

        volatile = False

        if not state['train']:
            volatile = True
        state['sample'][0] = Variable(data=state['sample'][0], volatile=volatile)
        state['sample'][1] = Variable(data=state['sample'][1], volatile=volatile)

    def save_model(state, filename):
        model = state['network']
        torch.save({'model': copy.deepcopy(model).cpu().state_dict(), 'optimizer': state['optimizer'].state_dict()},
                   filename)
        print('==>Model {} saved.'.format(filename))

    def adjust_learning_rate(state):
        optimizer = state['optimizer']
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

        print('~~~~~~~~~~~~~~~~~~adjust learning rate~~~~~~~~~~~~~~~~~~~~')

    # ***************************Callback Setting******************************

    def on_start(state):
        # wrap network
        if state['gpu_ids'] is None:
            print('Training/Validating without gpus ...')
        else:
            if not torch.cuda.is_available():
                raise RuntimeError('Cuda is not available')

            state['network'].cuda(state['gpu_ids'][0])
            print('Training/Validating on gpu: {}'.format(state['gpu_ids']))

        if state['train']:
            print('*********************Start Training at {}***********************'.format(time.strftime('%c')))
            if state['t'] == 0:
                filename = os.path.join(checkpoint_save_path, 'init_model.pth.tar')
                save_model(state, filename)
        else:
            print('-------------Start Validation at {} For Epoch{}--------------'.format(time.strftime('%c'),
                                                                                         state['epoch']))
        prepare_network(state)
        reset_meters()

    def on_start_epoch(state):
        # change state of the network
        reset_meters()
        print('--------------Start Training at {} for Epoch{}-----------------'.format(time.strftime('%c'),
                                                                                       state['epoch']))
        time_meter.reset()
        prepare_network(state)

    def on_end_sample(state):
        # wrap data
        state['sample'].append(state['train'])
        wrap_data(state)
        meters['data_loading_meter'].add(time_meter.value())

    def on_start_forward(state):
        # timing
        time_meter.reset()

    def on_end_forward(state):
        # loss meters
        if state['train']:
            meters['train_loss_meter'].add(state['loss'].data[0])
        else:
            meters['test_loss_meter'].add(state['loss'].data[0])

    def on_end_update(state):
        # logging info and saving model
        meters['gpu_time_meter'].add(time_meter.value())
        if state['t'] % iter_log_freq == 0 and state['t'] != 0:
            meters['data_loading_meter'].log('data_t', x=state['t'])
            meters['gpu_time_meter'].log('gpu_t', x=state['t'])
            meters['train_loss_meter'].log('train_loss_iteration', x=state['t'])

        if checkpoint_iter_freq and state['t'] % checkpoint_iter_freq == 0:
            filename = os.path.join(checkpoint_save_path,
                                    'e' + str(state['epoch']) + 't' + str(state['t']) + '.pth.tar')
            save_model(state, filename)
        time_meter.reset()

    def on_end_epoch(state):
        # logging info and saving model

        meters['train_loss_meter'].log('train_loss_epoch', x=state['epoch'])
        print('***************Epoch {} done: loss {}*****************'.format(state['epoch'],
                                                                              meters['train_loss_meter'].meter.value()))
        if checkpoint_epoch_freq and state['epoch'] % checkpoint_epoch_freq == 0:
            filename = os.path.join(checkpoint_save_path,
                                    'e' + str(state['epoch']) + 't' + str(state['t']) + '.pth.tar')
            save_model(state, filename)

        # adjust learning rate
        if state['epoch'] in lr_points:
            adjust_learning_rate(state)

        reset_meters()

        # do validation at the end of epoch
        state['train'] = False
        engine.validate()
        state['train'] = True

    def on_end_test(state):
        # calculation
        meters['test_loss_meter'].log('test_loss', x=state['epoch'])
        print('----------------Test epoch {} done: loss {}------------------'.format(state['epoch'], meters[
            'test_loss_meter'].meter.value()))
        reset_meters()

    def on_end(state):
        # logging
        t = time.strftime('%c')
        if state['train']:
            print('*********************Training done at {}***********************'.format(t))
        else:
            print('*********************Validation done at {}***********************'.format(t))

    engine.hooks['on_start'] = on_start
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_sample'] = on_end_sample
    engine.hooks['on_start_forward'] = on_start_forward
    engine.hooks['on_end_forward'] = on_end_forward
    engine.hooks['on_end_update'] = on_end_update
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.hooks['on_end_test'] = on_end_test
    engine.hooks['on_end'] = on_end

    return engine
