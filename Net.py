import torch.nn as nn


class conv_block(nn.Module):
    def __init__(self, chann_in, chann_out, k_size, stride, p_size, dilation=1):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=chann_in, out_channels=chann_out, kernel_size=k_size, stride=stride, padding=p_size,
                      dilation=dilation),
            nn.BatchNorm3d(chann_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Tracker_Net(nn.Module):
    def __init__(self, n_actions=500):
        super().__init__()

        self.layer1 = conv_block(1, 32, 3, stride=1, p_size=0)
        self.layer2 = conv_block(32, 32, 3, stride=1, p_size=0)
        self.layer3 = conv_block(32, 32, 3, stride=1, p_size=0, dilation=2)
        self.layer4 = conv_block(32, 32, 3, stride=1, p_size=0, dilation=4)
        self.layer5 = conv_block(32, 64, 3, stride=1, p_size=0)
        self.layer6 = conv_block(64, 64, 1, stride=1, p_size=0)
        self.n_classes = n_actions
        self.layer7 = nn.Conv3d(64, n_actions + 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)

        return out


class Detector_Net(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.layer1 = conv_block(1, 32, 3, stride=1, p_size=0)
        self.layer2 = conv_block(32, 32, 3, stride=1, p_size=0)
        self.layer3 = conv_block(32, 32, 3, stride=1, p_size=0, dilation=2)
        self.layer4 = conv_block(32, 32, 3, stride=1, p_size=0, dilation=4)
        self.layer5 = conv_block(32, 64, 3, stride=1, p_size=0)
        self.layer6 = conv_block(64, 64, 1, stride=1, p_size=0)
        self.layer7 = nn.Conv3d(64, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        bifur_end_radi = self.layer7(out)
        return bifur_end_radi