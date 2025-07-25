import math
import numpy as np
import SimpleITK as sitk
import torch

from Net import *
from collections import deque
from utils import *

N_ACTIONS = 500

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





# 存储分支起点
bifur_queue = deque()

with torch.no_grad():
    tracker = Tracker_Net()
    tracker.load_state_dict(torch.load(r'/home/zyy/training/Tracker.pth'))
    tracker.to(device)
    tracker.eval()

    detector = Detector_Net()
    detector.load_state_dict(torch.load(r'/home/zyy/training/Detector.pth', map_location='cuda:1'))
    detector.to(device)
    detector.eval()

    seeds = np.load(r'/home/zyy/training/dataset00/vessel0/vox_radi.npy')  # z, y, x, radi
    seed1, seed2 = seeds[100, :3], seeds[110, :3]  # z, y, x
    start_direction = seed2 - seed1  # z, y, x

    CT_filename = r'/home/zyy/training/dataset00/image00.mhd'
    img = sitk.ReadImage(CT_filename)
    img_arr = sitk.GetArrayFromImage(img)
    Spacing = img.GetSpacing()
    direction_list = create_actions(N_ACTIONS, dis=2, spacing=Spacing)
    patch = clip_patch(img_arr, seed1, gap=9)
    patch = patch.astype(float)
    patch = torch.from_numpy(patch).type(torch.float32)
    # patch = patch.type(torch.float32)
    patch = patch.reshape(1, 1, 19, 19, 19)
    patch = patch.to(device)

    directions_prob = tracker(patch)
    directions_prob = torch.squeeze(directions_prob)
    directions_prob = torch.softmax(directions_prob, dim=0)
    end_entropy = normalized_entropy(directions_prob)


    bifur_end_radi = torch.squeeze(detector(patch))
    bifur_prob, end_prob, radi = torch.sigmoid(bifur_end_radi[0]), torch.sigmoid(bifur_end_radi[1]), bifur_end_radi[2]

    CT_z = np.array([1, 0, 0])  # CT图像矩阵的z轴方向
    previous_direction = start_direction
    p_current = seed1

    centerline_points = [seed1]
    radi_list = [radi]


    while bifur_queue != 0 or end_entropy < 0.9:
        if bifur_prob > 0.5:

            groups = detect_bsp(previous_direction, CT_z, p_current, img_arr,voxel_threshold=2600)
            p1, p2 = groups[0], groups[1]
            p1, p2 = np.round(p1).astype(int), np.round(p2).astype(int)
            bifur_queue.append((p_current, p1))
            bifur_queue.append((p_current, p2))
            last_current = bifur_queue.popleft()
            p_last = last_current[0]
            p_current = last_current[1]
            previous_direction = p_current - p_last
        elif end_entropy >= 0.9:
            if not bifur_queue:
                break
            p_current = bifur_queue.popleft()
        else:
            directions_prob = tracker(patch)
            directions_prob = torch.squeeze(directions_prob)
            directions_prob = torch.softmax(directions_prob, dim=0)

            end_entropy = normalized_entropy(directions_prob)

            values, indices = torch.topk(directions_prob, k=2)
            direc1, direc2 = direction_list[indices[0].item()], direction_list[indices[1].item()]
            forward_direction = select_forward_direction(direc1, direc2, previous_direction)
            # unit_vector = forward_direction / vector_length(forward_direction)
            # step_lenth = 2 * unit_vector
            p_next = p_current + 2*forward_direction
            previous_direction = p_next - p_current
            p_current = p_next
        p_current = p_current.astype(int)
        centerline_points.append(p_current)
        patch = clip_patch(img_arr, p_current, gap=9)
        patch = patch.astype(float)
        patch = torch.from_numpy(patch).type(torch.float32)
        # patch = patch.type(torch.float32)
        patch = patch.reshape(1, 1, 19, 19, 19)
        bifur_end_radi = torch.squeeze(detector(patch))
        bifur_prob, end_prob, radi = torch.sigmoid(bifur_end_radi[0]), torch.sigmoid(bifur_end_radi[1]), bifur_end_radi[2]
        radi_list.append(radi)
        print(p_current, len(centerline_points))

        if len(centerline_points)==150:
            break

centerline_points = np.array(centerline_points)
radi_list = np.array(radi_list)
np.savetxt(r'/home/zyy/training/dataset00/vessel0/CenterlineTreePoints.txt', centerline_points, fmt='%d')
np.savetxt(r'/home/zyy/training/dataset00/vessel0/RadiList.txt', radi_list, fmt='%3f')






