import math

import torch
import numpy as np
import SimpleITK as sitk
from branch_detec import bran_detec
import torch.nn as nn
# from ddqn import Net
# from detector import Detector


class Net(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=1, dilation=1),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, dilation=1),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, dilation=2),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1, dilation=4),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, dilation=1),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=1, stride=1, dilation=1),
            nn.ReLU()
        )

        self.out = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=6, kernel_size=1, stride=1, dilation=1),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        actions_value = self.out(x6)
        return actions_value


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
        output = output.view(-1, 3 * 1 * 1)

        return output

device = torch.device('cuda:1')
GAP = 9
vox_radi_ref = np.load(r'dataset00/vessel0/vox_radi_label.npy')
vox_ref = vox_radi_ref[:, :3]
s_root = np.array([vox_ref[0, 2], vox_ref[0, 1], vox_ref[0, 0]]) # z,y,x
path_mhd = r'dataset00/image00.mhd'
image = sitk.ReadImage(path_mhd)
img_arr = sitk.GetArrayFromImage(image)
radii = []
tree_pos = np.zeros((0,3))
s_root2 = s_root[np.newaxis, :]
tree_pos = np.append(tree_pos,s_root2, axis=0)
q = np.zeros((0,6)) # queue storing bifurcation-points
# 17 rays on hemisphere
rays = np.array(
    [[1, 0, 0], [0, 0, 1], [0, -1, 1], [0, 1, 0], [0, -1, -1], [0, 0, -1], [0, 1, -1], [0, 1, 0], [0, 1, 1],
     [1, 0, 1], [1, -1, 1], [1, -1, 0], [1, -1, -1], [1, 0, -1], [1, 1, -1], [1, 1, 0], [1, 1, 1]])


def clip_patch(img_arr, s):
    patch = img_arr[int(s[0])-GAP:int(s[0])+GAP+1, int(s[1])-GAP:int(s[1])+GAP+1, int(s[2])-GAP:int(s[2])+GAP+1]
    return patch

def subtrace(s, s_):
    pass

def choose_action(tracker, x):
    # x = torch.unsqueeze(torch.unsqueeze(x, 0), 0)
    # input only one sample
    actions_value = tracker(x)
    action = torch.max(actions_value, 1)[1].to('cpu').numpy()  # return the argmax index
    action = action[0][0][0][0]
    return action



def step(s, a):
    s_ = s
    if a == 0:
        s_ = np.array([s[0] + 2, s[1], s[2]])
    elif a == 1:
        s_ = np.array([s[0] - 2, s[1], s[2]])
    elif a == 2:
        s_ = np.array([s[0], s[1] - 2, s[2]])
    elif a == 3:
        s_ = np.array([s[0], s[1] + 2, s[2]])
    elif a == 4:
        s_ = np.array([s[0], s[1], s[2] - 2])
    elif a == 5:
        s_ = np.array([s[0], s[1], s[2] + 2])

    return s_


if __name__ == '__main__':
    with torch.no_grad():
        tracker = Net()
        tracker.load_state_dict(torch.load('eval_net.pkl'))
        tracker = tracker.to(device).eval()
        detector1 = Detector()
        detector1.load_state_dict(torch.load('detector.pkl'))
        detector1 = detector1.to(device).eval()

        patch = clip_patch(img_arr, s_root)
        patch = torch.from_numpy(patch.astype(np.float32)).to(device)
        patch = torch.unsqueeze(torch.unsqueeze(patch, 0), 0)
        a = choose_action(tracker, patch)
        s = step(s_root, a)
        s2 = s[np.newaxis, :]
        tree_pos = np.append(tree_pos,s2,axis=0)
        _s = s_root
        while True:
            patch = clip_patch(img_arr,s)
            patch = torch.from_numpy(patch.astype(np.float32)).to(device)
            patch = torch.unsqueeze(torch.unsqueeze(patch, 0), 0)
            radius, label1, label2 = detector1(patch).to('cpu').numpy()[0][0], torch.sigmoid(detector1(patch)).to('cpu').numpy()[0][1], torch.sigmoid(detector1(patch)).to('cpu').numpy()[0][2] # identify radius, bifurcation-point and endpoint

            if label1 > 0.5:
                p3, p4 = bran_detec(_s, s, rays, radius)
                sp3 = np.hstack(s, p3)
                sp4 = np.hstack(s, p4)
                q = np.append(q,sp3,axis=0)
                q = np.append(q,sp4,axis=0)
                tree_pos = np.append(tree_pos, p3, axis=0)
                tree_pos = np.append(tree_pos, p4, axis=0)

                _s = s
                s = q[0][3:6]
                q = np.delete(q, 0, axis=0)
            elif label2 > 0.5:
                if q.shape[0]>0:
                    _s = q[0][:3]
                    s = q[0][3:6]
                    q = np.delete(q, 0, axis=0)
                else:
                    break

            a = choose_action(tracker, patch)
            s_ = step(s, a)
            s_2 = s_[np.newaxis, :]
            tree_pos = np.append(tree_pos, s_2, axis=0)
            _s = s
            s = s_

        np.savetxt('cenpos.txt', tree_pos)







