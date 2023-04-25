import torch.nn as nn
import torch
import torch.nn.functional as F

def convT1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.ConvTranspose2d(out_planes, in_planes, kernel_size=1, stride=stride, bias=False)


def convT3x3(in_planes, out_planes, stride=1, padding_mode='zeros'):
    """3x3 convolution with padding"""
    if stride == 2:
        return nn.ConvTranspose2d(out_planes, in_planes, kernel_size=4, stride=stride, padding=1, bias=False, padding_mode=padding_mode)
    else :
        return nn.ConvTranspose2d(out_planes, in_planes, kernel_size=3, stride=stride, padding=1, bias=False, padding_mode=padding_mode)


class BasicDecBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, conv_pad_mode='zeros'):
        super(BasicDecBlock, self).__init__()
        
        self.conv1 = convT3x3(inplanes, planes, stride, conv_pad_mode)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv2 = convT3x3(planes, planes, padding_mode=conv_pad_mode)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU()
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv1(out)
        out = self.bn1(out)
        
        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)
    
        return out
    
class ResNetDecoder(nn.Module):
    def __init__(self, block, layers, num_input_channels, num_output_channels, img_size_x=32, img_size_y=40, zero_init_residual=False,
                conv_pad_mode='zeros'):
        super(ResNetDecoder, self).__init__()

        self.inplanes = 64

        self.conv1 = nn.ConvTranspose2d(64, num_input_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, conv_pad_mode=conv_pad_mode)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, conv_pad_mode=conv_pad_mode)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, conv_pad_mode=conv_pad_mode)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, conv_pad_mode=conv_pad_mode)

        #self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.conv_upsample = nn.ConvTranspose2d(512,512, kernel_size = (img_size_x//8,img_size_y//8), stride = 1, padding = 0, bias = False)
        self.fc = nn.Linear(num_output_channels, 512 * block.expansion,)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicDecBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, conv_pad_mode='zeros'):
        upsample = None
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                convT3x3(self.inplanes, planes *block.expansion, stride = 2),
                nn.BatchNorm2d(self.inplanes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, upsample, conv_pad_mode))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_pad_mode=conv_pad_mode))
        layers.reverse()
        return nn.Sequential(*layers)

    def forward(self, x):
        
        x = self.fc(x)
        x = x.view(-1,512,1,1)
        x = self.conv_upsample(x)

        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)

        x = self.conv1(x)
        #x = self.bn1(x)
        #x = self.relu(x)
        return x

def resnet18decoder(**kwargs):
    """Constructs a ResNet-18 model feature extractor.
    """
    return ResNetDecoder(BasicDecBlock, [2, 2, 2, 2], **kwargs)
