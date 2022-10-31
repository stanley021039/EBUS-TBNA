import torch
import torch.nn as nn
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
    def __init__(self, num_classes=1, init_weights=False):
        super(CNNLSTM, self).__init__()
        self.resnet = resnet101(pretrained=False)
        self.resnet.fc = nn.Sequential(nn.Linear(self.resnet.fc.in_features, 300))
        self.lstm = nn.LSTM(input_size=300, hidden_size=256, num_layers=3)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.out_act = nn.Sigmoid()

        if init_weights:
            initialize_weights(self)

    def forward(self, x_3d):
        self.lstm.flatten_parameters()
        B, T, C, H, W = x_3d.size()
        hidden = None
        x = x_3d.view(B * T, C, H, W)
        x = self.resnet(x)
        x = x.view(B, T, 300)
        for t in range(x.size(1)):
            out, hidden = self.lstm(x[:, t, :].unsqueeze(0), hidden)

        x = self.relu(self.fc1(out[-1, :, :]))
        x = self.out_act(self.fc2(x))
        return x


