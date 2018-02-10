import torchvision.models.resnet as resnet
import torch.nn as nn


class AtrousBottleneck(resnet.Bottleneck):
    expansion = 4

    def __init__(self, planes, rate):
        super(resnet.Bottleneck, self).__init__()
        inner_plans = int(planes/self.expansion)
        self.conv1 = nn.Conv2d(planes, inner_plans, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inner_plans)
        self.conv2 = nn.Conv2d(inner_plans, inner_plans, kernel_size=3, padding=rate, bias=False, dilation=rate)
        self.bn2 = nn.BatchNorm2d(inner_plans)
        self.conv3 = nn.Conv2d(inner_plans, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None


class DeepLabV3(resnet.ResNet):
    def __init__(self, layers, num_blocks, multigrid, num_classes=21):
        super(DeepLabV3, self).__init__(resnet.Bottleneck, layers, num_classes)
        del self.layer4, self.avgpool, self.fc

        fulconv = []
        for i in range(4, num_blocks + 1):
            base_rate = 2 ** (i - 3)
            rates = [base_rate * grid for grid in multigrid]
            fulconv.append(self._make_atrous_layer(planes=1024, rates=rates, blocks=layers[3]))

        self.fulconv = nn.Sequential(*fulconv)
        self.logits = nn.Conv2d(1024, num_classes, 1)

    def _make_atrous_layer(self, planes, rates, blocks):
        layers = []
        for i in range(blocks):
            layers.append(AtrousBottleneck(planes, rates[i]))

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


def resnet50(num_blocks, num_classes=21):
    model = DeepLabV3([3, 4, 6, 3], num_blocks, (1, 2, 1), num_classes)

    return model


def resnet101(num_blocks, num_classes=21):
    model = DeepLabV3([3, 4, 23, 3], num_blocks, (1, 2, 1), num_classes)

    return model
