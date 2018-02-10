from . import deeplab
import torch.utils.model_zoo as model_zoo
from utilities import initialization

model_urls = {'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth', }


def parse_model(model_name, num_classes):
    model = None
    if 'deeplabv3' in model_name:
        if model_name == 'deeplabv3-resnet50-4blks':
            model = deeplab.resnet50(4, num_classes)
        elif model_name == 'deeplabv3-resnet50-6blks':
            model = deeplab.resnet50(6, num_classes)

    else:
        raise RuntimeError('Model name not defined!')

    return model


def construct_model(args, resume_model=None):
    model = parse_model(args.model_name, args.num_classes)
    if args.transfer_learning:
        for m in model_urls:
            if m in args.model_name:
                initialization.load_partial_network(model, model_zoo.load_url(model_urls[m]))
                initialization.init_network(model.fulconv)
                initialization.init_network(model.logits)

    elif resume_model is not None:
        model.load_state_dict(resume_model)
        print('==> Resume model from {}.'.format(args.resume_model))
    else:
        initialization.init_network(model)

    return model


def construct_test_model(args, resume_model):
    model = parse_model(args.model_name, args.num_classes)
    model.load_state_dict(resume_model)
    print('==> Resume model from {}.'.format(args.resume_model))

    return model
