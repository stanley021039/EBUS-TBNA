import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from torchvision.models import resnet18, resnet101


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


class CNNLSTM(nn.Module):
    def __init__(self):
        super(CNNLSTM, self).__init__()
        self.resnet = resnet101(pretrained=False)
        self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, 300))
        self.lstm = nn.LSTM(input_size=300, hidden_size=256, num_layers=3)

    def forward(self, x_3d):
        self.lstm.flatten_parameters()
        B, T, C, H, W = x_3d.size()
        hidden = None
        x = x_3d.view(B * T, C, H, W)
        x = self.resnet(x)
        x = x.view(B, T, 300)
        for t in range(x.size(1)):
            out, hidden = self.lstm(x[:, t, :].unsqueeze(0), hidden)

        x = out[-1, :, :]
        return x


class Resnet_E(nn.Module):
    def __init__(self, Elayertofc=False):
        super(Resnet_E, self).__init__()
        self.Elayertofc = Elayertofc

        self.conv2D = nn.Conv2d(3, 16, kernel_size=5, padding=0, bias=False)
        if self.Elayertofc:
            self.Efc1 = nn.Linear(10000, 1000, bias=False)
            self.EBN1 = nn.ReLU()
            self.Efc2 = nn.Linear(1000, 1000, bias=False)
            self.EBN2 = nn.ReLU()
            self.Efc3 = nn.Linear(1000, 1000, bias=False)
        else:
            self.resnet = resnet101(pretrained=False)
            self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, 256))

    def forward(self, e):
        if self.Elayertofc:
            e = self.Efc1(e)
            e = self.EBN1(e)
            e = self.Efc2(e)
            e = self.EBN2(e)
            e = self.Efc3(e)
        else:
            e = self.resnet(e)

        return e


class CNNLSTMUDE(nn.Module):
    def __init__(self, num_classes=1, init_weights=False):
        super(CNNLSTMUDE, self).__init__()
        self.UD_path = CNNLSTM()  # 256
        self.E_path = Resnet_E()  # 256

        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

        if num_classes == 1:
            self.out_act = nn.Sigmoid()
        else:
            self.out_act = nn.Softmax(dim=1)

        if init_weights:
            initialize_weights(self)

    def forward(self, x, e):

        x = self.UD_path(x)
        e = self.E_path(e)

        out = torch.cat((x, e), dim=1)
        out = self.relu(self.fc1(out))
        out = self.out_act(self.fc2(out))

        return out