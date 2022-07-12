import torch
import numpy as np
import torch
import torch.nn as nn
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
import math
import time
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optim
device = torch.device('cuda:1')


gap = 9
BATCH_SIZE = 64
epoch = 40000
LR = 0.0005
ct_arr = np.load(r'ct_arr.npy')
# vox_radi_lab = np.load(r'vox_radi_label.npy')


def clip_patch(img_arr, s, gap):
    patch = img_arr[int(s[0]-gap):int(s[0]+gap+1), int(s[1]-gap):int(s[1]+gap+1), int(s[2]-gap):int(s[2]+gap+1)]
    patch = patch
    return patch


class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=1, dilation=1),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, dilation=1),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, dilation=2),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, dilation=4),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, dilation=1),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=1, stride=1, dilation=1),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=3, kernel_size=1, stride=1, dilation=1),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        output = self.out(x6)
        output = output.view(-1, 3*1*1)
        return output


class MyDataset(Dataset):
    def __init__(self, names_file, transform=None):
        super().__init__()
        self.transform = transform
        self.file = np.load(names_file)
        self.size = self.file.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        s = [int(self.file[idx][0]), int(self.file[idx][1]), int(self.file[idx][2])]
        patch = clip_patch(ct_arr, s, gap)
        label1, label2, label3 = self.file[idx][3], self.file[idx][4], self.file[idx][5]

        sample = {'patch': patch, 'label1': label1, 'label2': label2, 'label3': label3}
        if self.transform:
            sample = self.transform(sample)

        return sample


train_dataset = MyDataset(names_file='vox_radi_label.npy', transform=None)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
detector = Detector().to(device)
detector.train()
optimizer = optim.Adam(detector.parameters(), lr=LR)
loss_func1 = nn.MSELoss()
loss_func2 = nn.BCEWithLogitsLoss()
loss_func3 = nn.BCEWithLogitsLoss()


for epoch_i in range(epoch):
    for step, data in enumerate(train_loader):
        x, y1, y2, y3 = data['patch'], data['label1'], data['label2'], data['label3']
        y1, y2, y3 = y1.float(), y2.float(), y3.float()
        x, y1, y2, y3 = x.to(device), y1.to(device), y2.to(device), y3.to(device)
        x = torch.unsqueeze(x, 1)
        x = x.float()
        output = detector(x)
        output_0, output_1, output_2 = output[:, 0], torch.sigmoid(output[:, 1]), torch.sigmoid(output[:, 2])

        loss = loss_func1(output[:, 0], y1) + loss_func2(output[:, 1], y2) + loss_func3(output[:, 2], y3)
        print('loss:', loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('epoch_i', epoch_i)
torch.save(detector.state_dict(), 'detector.pkl')








