from . import deeplabv3


def construct_model(args, resume_model=None):
    if args.model_name == 'deeplabv3-resnet50-4blks':
        model = deeplabv3.resnet50(4, args.num_classes)
    elif args.model_name == 'deeplabv3-resnet50-6blks':
        model = deeplabv3.resnet50(6, args.num_classes)
    else:
        raise RuntimeError('Model name not defined!')
    if resume_model is not None:
        model.load_state_dict(resume_model)
        print('==> Resume model from {}.'.format(args.resume_model))
    else:
        print('==> Model not init!')

    return model
