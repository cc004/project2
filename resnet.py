from torch import nn, Tensor
import torch
from torchvision.models.utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.init = False

    def all_modules(self):
        yield self.conv_bn1
        yield self.conv_bn2
        yield self

    def forward(self, x: Tensor) -> Tensor:

        if not self.init:
            self.init = True
            self.conv_bn1 = torch.nn.utils.fusion.fuse_conv_bn_eval(self.conv1, self.bn1)
            self.conv_bn2 = torch.nn.utils.fusion.fuse_conv_bn_eval(self.conv2, self.bn2)
            self.conv1 = self.conv2 = self.bn1 = self.bn2 = None

        identity = x

        out = self.conv_bn1(x)
        out = self.relu(out)
        out = self.conv_bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(
        self,
        layers: List[int],
        num_classes: int = 1000,
    ) -> None:
        super(ResNet, self).__init__()

        self.inplanes = 64

        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, layers[0], 1)
        self.layer2 = self._make_layer(128, layers[1], 2)
        self.layer3 = self._make_layer(256, layers[2], 2)
        self.layer4 = self._make_layer(512, layers[3], 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        self.init = False

    def all_modules(self):
        def list_module(module):
            for m in module:
                yield from m.all_modules()
        yield self.conv_bn1
        yield from list_module(self.layer1)
        yield from list_module(self.layer2)
        yield from list_module(self.layer3)
        yield from list_module(self.layer4)
        yield self.avgpool
        yield self.fc
    
    def _make_layer(self, planes: int, blocks: int,
                    stride) -> nn.Sequential:
        downsample = None

        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * BasicBlock.expansion, stride),
                nn.BatchNorm2d(planes * BasicBlock.expansion),
            )

        layers = []

        layers.append(BasicBlock(self.inplanes, planes, stride, downsample))

        self.inplanes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        if not self.init:
            self.init = True
            self.conv_bn1 = torch.nn.utils.fusion.fuse_conv_bn_eval(self.conv1, self.bn1)
            self.conv1 = self.bn1 = None
        
        x = self.conv_bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

def new_model():
    return ResNet([2, 2, 2, 2], 10).cuda()
