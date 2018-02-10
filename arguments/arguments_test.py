from .arguments_base import ArgumentsBase


class ArgumentsTest(ArgumentsBase):
    def __init__(self):
        super(ArgumentsTest, self).__init__()

        parser = self.parser

        # the data set
        parser.add_argument('-dir', '--directory', required=True, help='test data directory',
                            metavar='DIR')
        # pre-processing args

        # model info
        parser.add_argument('-tm', '--resume-model', required=True, help='test model file', metavar='FILE')

        # log info
        parser.add_argument('-lp', '--logpath', required=True, metavar='DIR')
