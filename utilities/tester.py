import time

import torch
import torch.nn
import torch.nn.functional as functional
import torchnet.meter as meter
from torch.autograd import Variable
from tqdm import tqdm


def test(model, gpu_ids, iterator, topk, num_classes, enviroment='main'):
    print('=========================Start Testing at {}==========================='.format(time.strftime('%c')))

    classerr_meters = meter.ClassErrorMeter(topk)

    # multiple gpu support
    if gpu_ids is not None:
        model.cuda(gpu_ids[0])
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    # set eval() to freeze running mean and running var
    model.eval()

    for sample in tqdm(iterator):
        # wrap data
        for i in range(2):
            if gpu_ids is not None:
                sample[i].cuda(gpu_ids[0], async=True)
            sample[i] = Variable(sample[i], volatile=True)

        ipt, target = sample[0], sample[1]
        opt = functional.softmax(model(ipt), dim=1)

        classerr_meters.add(opt.data, target.data)

    classerrs = classerr_meters.value()
    print('========================Testing Down at {} ==========================='.format(time.strftime('%c')))
    print('******************Top {} Error: {}*****************'.format(topk, classerrs))
