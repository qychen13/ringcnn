import torch.nn.functional as f


class SoftMax2D(object):
    def __init__(self, size_average=True, ignore_index=255):
        self.size_average = size_average
        self.ignore_index = ignore_index

    def __call__(self, pdt, lbl):
        pdt = f.log_softmax(pdt, dim=1)
        pdt = f.upsample(pdt, size=lbl.shape[1:], mode='bilinear')
        return f.nll_loss(pdt, lbl, size_average=self.size_average, ignore_index=self.ignore_index)