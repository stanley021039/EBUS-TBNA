import sys
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=(1, 1, 1), downsample=None):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.BN1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.BN2 = nn.BatchNorm3d(out_channels)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.BN1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.BN2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class _EncoderBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, dsample, pool_kernal=(2, 2, 1), dropout=False):
        super(_EncoderBlock3D, self).__init__()
        if not dsample:
            layers = [
                BasicBlock3D(in_channels, out_channels)
            ]
        else:
            downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels * BasicBlock3D.expansion, kernel_size=1, stride=pool_kernal),
                nn.BatchNorm3d(out_channels * BasicBlock3D.expansion)
            )
            layers = [
                BasicBlock3D(in_channels, out_channels, stride=pool_kernal, downsample=downsample),
            ]

        if dropout:
            layers.append(nn.Dropout(p=0.5))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class BasicBlock2D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.BN1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.BN2 = nn.BatchNorm2d(out_channels)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.BN1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.BN2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class _EncoderBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, dsample, pool_kernal=2, dropout=False):
        super(_EncoderBlock2D, self).__init__()
        if not dsample:
            layers = [
                BasicBlock2D(in_channels, out_channels)
            ]
        else:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock2D.expansion, kernel_size=1, stride=pool_kernal),
                nn.BatchNorm2d(out_channels * BasicBlock2D.expansion)
            )
            layers = [
                BasicBlock2D(in_channels, out_channels, stride=pool_kernal, downsample=downsample),
            ]

        if dropout:
            layers.append(nn.Dropout(p=0.5))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class R3D_UDE(nn.Module):
    def __init__(self, num_classes, init_weights=False, time_steps=24, Elayertofc=False, mode='after', f_cat=False, o_act=True):
        super(R3D_UDE, self).__init__()
        self.mode = mode
        self.f_cat = f_cat
        self.o_act = o_act
        self.Elayertofc = Elayertofc
        self.conv3D = nn.Conv3d(3, 16, kernel_size=5, padding=2, bias=False)
        self.layer1 = _EncoderBlock3D(16, 16, dsample=True)
        self.layer2 = _EncoderBlock3D(16, 32, dsample=True)
        self.layer3 = _EncoderBlock3D(32, 64, dsample=True)
        self.layer4 = _EncoderBlock3D(64, 128, dsample=True)
        self.layer5 = _EncoderBlock3D(128, 256, dsample=True)
        self.layer6 = _EncoderBlock3D(256, 512, dsample=True)
        self.avg_pool = nn.AvgPool3d(kernel_size=(3, 3, time_steps), stride=2)
        if not f_cat:
            self.fc1 = nn.Linear(10240, 1000, bias=False)
        else:
            self.fc1 = nn.Linear(10240, 500, bias=False)
        self.fc2 = nn.Linear(1000, 512, bias=False)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(512, num_classes)

        self.conv2D = nn.Conv2d(3, 16, kernel_size=5, padding=0, bias=False)
        if self.Elayertofc:
            self.Efc1 = nn.Linear(10000, 1000, bias=False)
            self.EBN1 = nn.ReLU()
            self.Efc2 = nn.Linear(1000, 1000, bias=False)
            self.EBN2 = nn.ReLU()
            self.Efc3 = nn.Linear(1000, 1000, bias=False)
        else:
            self.elayer1 = _EncoderBlock2D(16, 16, dsample=True)
            self.elayer2 = _EncoderBlock2D(16, 32, dsample=True)
            self.elayer3 = _EncoderBlock2D(32, 64, dsample=True)
            self.elayer4 = _EncoderBlock2D(64, 128, dsample=True)
            self.elayer5 = _EncoderBlock2D(128, 256, dsample=True)
            self.elayer6 = _EncoderBlock2D(256, 512, dsample=True)
            self.eavg_pool = nn.AvgPool2d(kernel_size=3, stride=2)
            if not f_cat:
                self.efc = nn.Linear(10240, 1000, bias=False)
            else:
                self.efc = nn.Linear(10240, 500, bias=False)
        if self.mode:
            self.fc_atten = nn.Linear(2, 1000, bias=False)

        if self.o_act:
            if num_classes == 1:
                self.out_act = nn.Sigmoid()
            else:
                self.out_act = nn.Softmax(dim=1)

        if init_weights:
            initialize_weights(self)

    def forward(self, x, e, graph):
        batch_size, C, H, W, time_depth = x.size()
        x = self.conv3D(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.avg_pool(x)
        out = self.fc1(x.view(batch_size, -1))

        if self.Elayertofc:
            e = self.Efc1(e)
            e = self.EBN1(e)
            e = self.Efc2(e)
            e = self.EBN2(e)
            e = self.Efc3(e)
        else:
            e = self.conv2D(e)
            e = self.elayer1(e)
            e = self.elayer2(e)
            e = self.elayer3(e)
            e = self.elayer4(e)
            e = self.elayer5(e)
            e = self.elayer6(e)
            e = self.eavg_pool(e)
            e = self.efc(e.view(batch_size, -1))

        if self.mode == 'before':
            attention = self.fc_atten(graph)
            out = out * attention
            if not self.f_cat:
                out = out + e
            else:
                out = torch.cat((out, e), 1)
        elif self.mode == 'after':
            if not self.f_cat:
                out = out + e
            else:
                out = torch.cat((out, e), 1)
            attention = self.fc_atten(graph)
            out = out * attention

        out2 = self.fc2(out)
        out2 = self.relu(out2)
        out3 = self.fc3(out2)
        if self.o_act:
            out3 = self.out_act(out3)
        return out3


class R3D_UD(nn.Module):
    def __init__(self, num_classes, init_weights=False, time_steps=24, mode_signal=True, out_act=True):
        super(R3D_UD, self).__init__()
        self.mode_signal = mode_signal
        self.out_act = out_act
        self.conv3D = nn.Conv3d(3, 16, kernel_size=5, padding=2, bias=False)
        self.layer1 = _EncoderBlock3D(16, 16, dsample=True)
        self.layer2 = _EncoderBlock3D(16, 32, dsample=True)
        self.layer3 = _EncoderBlock3D(32, 64, dsample=True)
        self.layer4 = _EncoderBlock3D(64, 128, dsample=True)
        self.layer5 = _EncoderBlock3D(128, 256, dsample=True)
        self.layer6 = _EncoderBlock3D(256, 512, dsample=True)
        self.avg_pool = nn.AvgPool3d(kernel_size=(3, 3, time_steps), stride=2)
        self.fc1 = nn.Linear(10240, 1000, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1000, num_classes, bias=False)

        if self.mode_signal:
            self.fc_atten = nn.Linear(2, 1000, bias=False)

        if self.out_act:
            if num_classes == 1:
                self.o_act = nn.Sigmoid()
            else:
                self.o_act = nn.Softmax(dim=1)

        if init_weights:
            initialize_weights(self)

    def forward(self, x, graph=None):
        batch_size, C, H, W, time_depth = x.size()
        x = self.conv3D(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.avg_pool(x)
        out_f = self.fc1(x.view(batch_size, -1))

        if self.mode_signal:
            attention = self.fc_atten(graph)
            out_f = out_f * attention

        out_1 = self.relu(out_f)
        out_2 = self.fc2(out_1)

        if self.out_act:
            out_2 = self.o_act(out_2)

        return out_2
