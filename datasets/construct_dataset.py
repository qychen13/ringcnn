import torch.cuda
from torch.utils.data import DataLoader

from .voc import dataset as vocdataset


def construct_train_dataloaders(args):
    train_sampler = None
    val_sampler = None
    shuffle = True

    # ********************************** Dataset Setup *********************************

    train_folder = vocdataset.SBDClassSeg(args.directory, split='train')
    val_folder = vocdataset.SBDClassSeg(args.directory, split='val')

    # ********************************** Wrap Data Loader ********************************
    pin_memory = False
    if torch.cuda.is_available():
        pin_memory = True
    train_data_loader = DataLoader(train_folder, args.batch_size, shuffle=shuffle, sampler=train_sampler,
                                   num_workers=args.num_workers, pin_memory=pin_memory, drop_last=False)
    val_data_loader = DataLoader(val_folder, args.batch_size, shuffle=False, sampler=val_sampler,
                                 num_workers=args.num_workers, pin_memory=pin_memory, drop_last=False)

    return train_data_loader, val_data_loader


def construct_test_dataloaders(args):
    test_folder = vocdataset.VOC2012ClassSeg(args.directory, split='val')

    pin_memory = False
    if torch.cuda.is_available():
        pin_memory = True

    test_dataloader = DataLoader(test_folder, args.batch_size, shuffle=False, num_workers=args.num_workers,
                                 pin_memory=pin_memory)

    return test_dataloader
