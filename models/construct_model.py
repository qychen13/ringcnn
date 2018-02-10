from . import deeplabv3
import torch.utils.model_zoo as model_zoo
from utilities import initialization

models = {'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth', }


def construct_model(args, resume_model=None):
    if 'deeplabv3' in args.model_name:
        if args.model_name == 'deeplabv3-resnet50-4blks':
            model = deeplabv3.resnet50(4, args.num_classes)
        elif args.model_name == 'deeplabv3-resnet50-6blks':
            model = deeplabv3.resnet50(6, args.num_classes)
    else:
        raise RuntimeError('Model name not defined!')
    if args.transfer_learning:
        for m in models:
            if m in args.model_name:
                initialization.load_partial_network(model, model_zoo.load_url(models[m]))
                initialization.init_network(model.fulconv)
                initialization.init_network(model.logits)

    elif resume_model is not None:
        model.load_state_dict(resume_model)
        print('==> Resume model from {}.'.format(args.resume_model))
    else:
        initialization.init_network(model)

    return model
