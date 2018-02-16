import time
import numpy as np
import os

import torch
import torch.nn
import torch.nn.functional
import torchnet.meter as meter
from torch.autograd import Variable
from tqdm import tqdm

import utilities.segmentation_meter


def test(model, gpu_ids, iterator, num_classes, logpath):
    print('=========================Start Testing at {}==========================='.format(time.strftime('%c')))

    confusion_meter = meter.ConfusionMeter(num_classes)

    # multiple gpu support
    if gpu_ids is not None:
        model.cuda(gpu_ids[0])
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    # set eval() to freeze running mean and running var
    model.eval()

    ignore_lbl = iterator.dataset.ignore_lbl
    for sample in tqdm(iterator):
        # wrap data
        for i in range(2):
            if gpu_ids is not None:
                sample[i] = sample[i].cuda(gpu_ids[0], async=True)
                sample[i] = Variable(sample[i], volatile=True)

        ipt, target = sample[0], sample[1]
        opt = model(ipt)

        # compute confusion matrix
        opt, target=utilities.segmentation_meter.preprocess_for_confusion(opt, target, ignore_lbl)
        confusion_meter.add(opt, target)

    confusion_matrix = confusion_meter.value()
    np.savetxt(os.path.join(logpath, 'confution_matrix.csv'), confusion_matrix, delimiter=',')

    meters = utilities.segmentation_meter.compute_segmentation_meters(confusion_matrix)

    print('========================Testing Down at {} ==========================='.format(time.strftime('%c')))
    print('******************pixel acc.: {pacc}, mean acc: {macc}, mIU: {miu}, f.w.IU:{fwiu} '
          '*****************'.format(**meters))
