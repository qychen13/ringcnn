"""Motivatied by torchvison.resnet
"""
import torchvision.models.resnet as resnet
import torch.nn as nn


class DeepLabBase(resnet.ResNet):
    def __init__(self, layers, num_classes=21):
        super(DeepLabBase, self).__init__(resnet.Bottleneck, layers, num_classes)
        del self.layer4, self.avgpool, self.fc
        self.fulconv = None
        self.logits = None

    def _make_fulconv_layers(self, block, block_args_list):
        layers = []
        for args in block_args_list:
            layers.append(block(*args))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.fulconv(x)
        x = self.logits(x)

        return x


class AtrousBottleneck(resnet.Bottleneck):
    expansion = 4

    def __init__(self, planes, rate):
        super(resnet.Bottleneck, self).__init__()
        inner_plans = int(planes / self.expansion)
        self.conv1 = nn.Conv2d(planes, inner_plans, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inner_plans)
        self.conv2 = nn.Conv2d(inner_plans, inner_plans, kernel_size=3, padding=rate, bias=False, dilation=rate)
        self.bn2 = nn.BatchNorm2d(inner_plans)
        self.conv3 = nn.Conv2d(inner_plans, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None


class DeepLabAtrous(DeepLabBase):
    def __init__(self, block, layers, num_blocks, multigrid, num_classes=21):
        super(DeepLabAtrous, self).__init__(layers, num_classes)

        fulconv = []
        for i in range(4, num_blocks + 1):
            base_rate = 2 ** (i - 3)
            rates = [base_rate * grid for grid in multigrid]
            block_args_list = [[1024, rate] for rate in rates]
            print(block_args_list)
            fulconv.append(self._make_fulconv_layers(block, block_args_list))

        self.fulconv = nn.Sequential(*fulconv)
        self.logits = nn.Conv2d(1024, num_classes, 1)


def deeplab(version, model_name, num_blocks, num_classes):
    layers = {'resnet50': [3, 4, 6, 3], 'resnet101': [3, 4, 23, 3]}

    args = dict(layers=layers[model_name], num_blocks=num_blocks, num_classes=num_classes)

    if version == 1:
        args['multigrid'] = (1, 1, 1)
        args['block'] = AtrousBottleneck
    elif version == 2:
        args['multigrid'] = (1, 2, 1)
        args['block'] = AtrousBottleneck
    else:
        raise RuntimeError('DeepLab version not support!')

    return DeepLabAtrous(**args)


def resnet50(num_blocks, num_classes=21):
    return deeplab(2, 'resnet50', num_blocks, num_classes)


