from arguments.arguments_test import ArgumentsTest
from models.construct_model import construct_model
from utilities.tester import test
from datasets.construct_dataset import construct_test_dataloaders
import torch


def main():
    args = ArgumentsTest().parse_args()

    print('***************************Arguments****************************')
    print(args)

    checkpoint = torch.load(args.resume_model)
    model = construct_model(args, checkpoint['model'])

    print('==> Resume Model {}'.format(args.resume_model))

    print('--------------------------Model Info----------------------------')
    print(model)

    topk = [1, 5]

    test_iterator = construct_test_dataloaders(args)
    # _ ,test_iterator = construct_train_dataloaders(args)

    test(model, args.gpu_ids, iterator=test_iterator, topk=topk, num_classes=args.num_classes,
         enviroment=args.environment)


if __name__ == '__main__':
    main()
