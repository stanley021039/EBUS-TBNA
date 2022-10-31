#import sys
import torch
#import torch.nn.functional as F
from torch import nn
#import numpy as np

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


class SEBlock(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(SEBlock, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, mid_channels)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(mid_channels, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        bs, c, h, w = x.size()
        z = self.avgpool(x)
        z = self.fc1(z.view(bs, -1))
        z = self.relu(z)
        z = self.fc2(z)
        z = self.sigmoid(z)
        z = z.view(bs, c, 1, 1)
        return x * z.expand_as(x)

class FireBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, SE_channels):
        super(FireBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.conv21 = nn.Conv2d(mid_channels, in_channels, 1)
        self.conv22 = nn.Conv2d(mid_channels, in_channels, 3, padding=1)
        self.SE = SEBlock(out_channels, SE_channels)

    def forward(self, x):
        z1 = self.conv1(x)
        z21 = self.conv21(z1)
        z22 = self.conv22(z1)
        z21 = x + z21
        z22 = x + z22
        z3 = torch.cat([z21, z22], 1)
        return self.SE(z3)

class FCentralBox(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, SE_channels):
        super(FCentralBox, self).__init__()
        self.fireG = FireBlock(in_channels, mid_channels, out_channels, SE_channels)
        self.fireF = FireBlock(in_channels, mid_channels, out_channels, SE_channels)
        self.fireE = FireBlock(in_channels, mid_channels, out_channels, SE_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cat_weight_G = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.cat_weight_F = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.cat_weight_E = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

        self.cat_weight_G.data.fill_(0.33)
        self.cat_weight_F.data.fill_(0.33)
        self.cat_weight_E.data.fill_(0.33)

    def forward(self, G, F, E):
        G = self.maxpool(self.fireG(G))
        F = self.maxpool(self.fireG(F))
        E = self.maxpool(self.fireG(E))
        z4 = torch.cat([self.cat_weight_G * G, self.cat_weight_F * F, self.cat_weight_E * E], dim=1)

        return G, F, E, z4

class OCentralBox(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, SE_channels):
        super(OCentralBox, self).__init__()
        self.fire1 = FireBlock(in_channels, mid_channels, out_channels, SE_channels)
        self.fire2 = FireBlock(in_channels, mid_channels, out_channels, SE_channels)
        self.fire3 = FireBlock(in_channels, mid_channels, out_channels, SE_channels)
        self.fire4 = FireBlock(in_channels * 3, mid_channels * 3, out_channels * 3, SE_channels * 3)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cat_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.cat_weight_2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.cat_weight_3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.sum_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.sum_weight_2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

        self.cat_weight_1.data.fill_(0.33)
        self.cat_weight_2.data.fill_(0.33)
        self.cat_weight_3.data.fill_(0.33)
        self.sum_weight_1.data.fill_(0.5)
        self.sum_weight_2.data.fill_(0.5)

    def forward(self, x1, x2, x3, x4):
        z1 = self.maxpool(self.fire1(x1))
        z2 = self.maxpool(self.fire2(x2))
        z3 = self.maxpool(self.fire3(x3))
        z4 = self.maxpool(self.fire4(x4))
        z_cat = torch.cat([self.cat_weight_1 * z1, self.cat_weight_2 * z2, self.cat_weight_3 * z3], dim=1)
        z4 = self.sum_weight_1 * z4 + self.sum_weight_2 * z_cat

        return z1, z2, z3, z4

class EBUSNet(nn.Module):
    def __init__(self, init_weights=False):
        super(EBUSNet, self).__init__()
        self.convG = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.convF = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.convE = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)

        self.central1 = FCentralBox(32, 16, 64, 32)
        self.central2 = OCentralBox(64, 32, 128, 64)
        self.central3 = OCentralBox(128, 64, 256, 128)
        self.central4 = OCentralBox(256, 128, 512, 256)

        self.FC1 = nn.Linear(25088, 1)
        self.FC2 = nn.Linear(25088, 1)
        self.FC3 = nn.Linear(25088, 1)
        self.FC4 = nn.Linear(75264, 1)

        self.sigmoid = nn.Sigmoid()

        if init_weights:
            initialize_weights(self)

    def forward(self, G, F, E):
        bs, c, w, h = G.size()
        G = self.convG(G)
        F = self.convF(F)
        E = self.convE(E)
        z1, z2, z3, z4 = self.central1(G, F, E)
        z1, z2, z3, z4 = self.central2(z1, z2, z3, z4)
        z1, z2, z3, z4 = self.central3(z1, z2, z3, z4)
        z1, z2, z3, z4 = self.central4(z1, z2, z3, z4)
        out1 = self.FC1(z1.view(bs, -1))
        out2 = self.FC2(z2.view(bs, -1))
        out3 = self.FC3(z3.view(bs, -1))
        out4 = self.FC4(z4.view(bs, -1))
        return self.sigmoid(out1 + out2 + out3 + out4)
