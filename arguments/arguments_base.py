import argparse


class ArgumentsBase(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        parser = self.parser
        # gpu id
        parser.add_argument('-gs', '--gpu-ids', type=int, nargs='+',
                            help='multiple gpu device ids to train the network')

        # the data set
        parser.add_argument('-b', '--batch-size', required=True, type=int, help='mini-batch size')
        parser.add_argument('-nw', '--num-workers', default=4, type=int, help='workers for loading data')

        # logfile info
        parser.add_argument('-en', '--environment', type=str, default='main', help='log environment for visdom')

        # model info
        parser.add_argument('-model', '--model-name', type=str, required=True,
                            help='model name')
        parser.add_argument('-nc', '--num_classes', type=int, default=1000, help='number of the classes')

    def parse_args(self):
        return self.parser.parse_args()
