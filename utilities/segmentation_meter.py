import numpy as np
import torch.nn.functional


def compute_segmentation_meters(confusion_matrix):
    pacc = np.diag(confusion_matrix).sum() / confusion_matrix.sum()
    acc = np.diag(confusion_matrix) / confusion_matrix.sum(1)
    iu = np.diag(confusion_matrix) / (confusion_matrix.sum(1) + confusion_matrix.sum(0) - np.diag(confusion_matrix))
    freq = confusion_matrix.sum(1) / confusion_matrix.sum()
    meters = dict(miu=np.nanmean(iu), pacc=pacc, macc=np.nanmean(acc), fwiu=(freq[freq > 0] * iu[freq > 0]).sum())

    return meters


def preprocess_for_confusion(opt, target, ignore_lbl):
    num_classes = opt.shape[1]
    if opt.shape[2] != target.shape[1] or opt.shape[3] != target.shape[2]:
        opt = torch.nn.functional.upsample(opt, target.shape[1:], mode='bilinear')
    opt = opt.data
    target = target.data
    opt = opt.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
    target = target.view(-1)
    index = (target != ignore_lbl)
    opt = opt[index.unsqueeze(1).expand_as(opt)].view(-1, num_classes)
    target = target[index]

    return opt, target
