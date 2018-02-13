from .arguments_base import ArgumentsBase
import os

class ArgumentsTrainVal(ArgumentsBase):
    def __init__(self):
        super(ArgumentsTrainVal, self).__init__()

        parser = self.parser

        # the data set
        parser.add_argument('-dir', '--directory', default=os.environ['DATASET_DIR'], help='training and validation data directory',
                            metavar='DIR')
        # pre-processing args

        # log info
        parser.add_argument('-ilog', '--iter-log-freq', type=int, default=100, help='log frequency under iterations')

        # model save info
        parser.add_argument('-cifrec', '--checkpoint-iter-freq', default=5000, type=int,
                            help='the frequency of saving model under iteration')
        parser.add_argument('-cefrec', '--checkpoint-epoch-freq', default=5, type=int,
                            help='the frequency of saving model under epoch')
        parser.add_argument('-cpath', '--checkpoint_save-path', required=True, help='the directory to save model')

        # training control parameters
        parser.add_argument('-e', '--maxepoch', required=True, type=int, help='the number of epochs to train')
        parser.add_argument('-lr', '--learning-rate', default=0.1, type=float, help='initial learning rate')
        parser.add_argument('-m', '--momentum', default=0.9, type=float, help='momentum')
        parser.add_argument('-wd', '--weight-decay', default=1e-4, type=float, help='weight decay(L1 penalty)')

        # model info
        parser.add_argument('-tl', '--transfer-learning', default=0, type=int)
        parser.add_argument('-rm', '--resume-model', default=None, help='resume model file', metavar='FILE')
        parser.add_argument('-iter', '--resume-iteration', default=0, type=int, help='resume iteration number')
        parser.add_argument('-epo', '--resume-epoch', default=0, type=int, help='resume epoch number')

        # criterion info

        # dataset info
        parser.add_argument('-ds', '--dataset', type=str, help='dataset name used for training')

