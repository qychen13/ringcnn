import time
import numpy as np
import os

import torch
import torch.nn
import torchnet.meter as meter
from torch.autograd import Variable
from tqdm import tqdm


def test(model, gpu_ids, iterator, num_classes, logpath):
    print('=========================Start Testing at {}==========================='.format(time.strftime('%c')))

    confusion_meter = meter.ConfusionMeter(num_classes)

    # multiple gpu support
    if gpu_ids is not None:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    # set eval() to freeze running mean and running var
    model.eval()

    ignore_lbl = iterator.dataset.ignore_lbl
    for sample in tqdm(iterator):
        # wrap data
        for i in range(2):
            if gpu_ids is not None:
                sample[i].cuda(gpu_ids[0], async=True)
        sample[i] = Variable(sample[i], volatile=True)

        ipt, target = sample[0], sample[1]
        ipt = Variable(ipt, volatile=True)
        opt = model(ipt)

        # compute confusion matrix
        opt = opt.data
        opt = opt.permute(0, 2, 3, 1).view(-1, num_classes)
        target = target.view_as(opt)
        index = target != ignore_lbl
        opt = opt[index]
        target = target[target]
        confusion_meter.add(opt, target)

    confusion_matrix = confusion_meter.value()
    np.savetxt(os.path.join(logpath, 'confution_matrix.csv'), confusion_matrix, delimiter=',')

    pacc = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
    acc = np.diag(confusion_matrix) / confusion_matrix.sum(1)
    iu = np.diag(confusion_matrix) / (confusion_matrix.sum(1) + confusion_matrix.sum(0) - np.diag(confusion_matrix))
    freq = confusion_matrix.sum(1) / confusion_matrix.sum()

    print('========================Testing Down at {} ==========================='.format(time.strftime('%c')))
    print('******************pixel acc.: {}, mean acc: {}, mIU: {}, f.w.IU:{} '
          '*****************'.format(pacc, np.nanmean(acc), np.nanmean(iu), (freq[freq > 0] * iu[freq > 0]).sum()))
