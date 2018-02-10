from arguments.arguments_trainval import ArgumentsTrainVal
from models.construct_model import construct_model
from utilities.construct_engine import construct_engine
from datasets.construct_dataset import construct_train_dataloaders
import utilities.loss

from torch.optim.sgd import SGD
import torch


def main():
    # ============================= args setting =================================
    args = ArgumentsTrainVal().parse_args()

    print('***************************Arguments****************************')
    print(args)

    checkpoint = None
    resume_model = None
    resume_optimizer = None
    if args.resume_model is not None:
        checkpoint = torch.load(args.resume_model)
        if 'model' in checkpoint.keys:
            resume_model = checkpoint['model']
        else:
            resume_model = checkpoint
        if 'optimizer' in checkpoint.keys():
            resume_optimizer = checkpoint['optimizer']

    # ============================= model setting =================================

    model = construct_model(args, resume_model)
    print('--------------------------Model Info----------------------------')
    print(model)

    # ============================= training setting =================================

    train_iterator, validate_iterator = construct_train_dataloaders(args)

    criterion = utilities.loss.SoftMax2D(ignore_index=train_iterator.dataset.ignore_lbl)
    optimizer = SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    if resume_optimizer is not None:
        optimizer.load_state_dict(checkpoint[optimizer])

    engine_set = dict(gpu_ids=args.gpu_ids,
                      network=model,
                      criterion=criterion,
                      train_iterator=train_iterator,
                      validate_iterator=validate_iterator,
                      optimizer=optimizer)

    # learning rate points
    lr_points = []
    if args.num_classes == 100:
        lr_points = [150, 225]
    print('==> Set lr_points: {}'.format(lr_points))

    engine_args = dict(checkpoint_iter_freq=args.checkpoint_iter_freq,
                       checkpoint_epoch_freq=args.checkpoint_epoch_freq,
                       checkpoint_save_path=args.checkpoint_save_path, iter_log_freq=args.iter_log_freq,
                       environment=args.environment, lr_points=lr_points)

    engine = construct_engine(engine_set, **engine_args)

    engine.resume(args.maxepoch, args.resume_epoch, args.resume_iteration)


if __name__ == '__main__':
    main()
