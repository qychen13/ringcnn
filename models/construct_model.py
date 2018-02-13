from . import deeplab, ringcnn, pspnet
import torch.utils.model_zoo as model_zoo
from utilities import initialization

model_urls = {'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth', }


def parse_model(model_name, num_classes):
    # name convention: deeplabv3-resnet50-4blks-ring
    model = None
    model_names = model_name.split('-')

    if 'deeplabv3' == model_names[0]:
        num_blocks = int(model_names[2][0])
        if 'resnet50' == model_names[1]:
            model = deeplab.resnet50(num_blocks, num_classes)
        if 'ring' in model_name:
            model = ringcnn.ringcnn_deeplab(model, base_rate=2)
    elif 'pspnet' == model_names[0]:
        model = pspnet.pspnet(model_names[1], num_classes)
        if 'ring' in model_name:
            model = ringcnn.ringcnn_dilatedfcn(model, base_rate=2)
    elif 'dilatedFCN' == model_names[0]:
        model = pspnet.dilatedFCN(model_names[1], num_classes)
        if 'ring' in model_name:
            model = ringcnn.ringcnn_dilatedfcn(model, base_rate=2)
    else:
        raise RuntimeError('Model name not defined!')

    if 'nondilated' in model_name:
        ringcnn.delete_dilated_conv(model)

    return model


def construct_model(args, resume_model=None):
    model = parse_model(args.model_name, args.num_classes)
    if args.transfer_learning:
        for m in model_urls:
            if m in args.model_name:
                initialization.load_partial_network(model, model_zoo.load_url(model_urls[m]))
                if model.fulconv is not None:
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
