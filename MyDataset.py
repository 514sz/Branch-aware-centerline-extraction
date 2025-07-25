import os
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MyDataset(Dataset):
    def __init__(self, patch_dir, label_dir):
        self.patch_dir = patch_dir
        self.label_dir = label_dir
        self.patch_list = os.listdir(self.patch_dir)
        self.label_list = os.listdir(self.label_dir)
    def __getitem__(self, item):
        patch_name = self.patch_list[item]
        patch_item_path = os.path.join(self.patch_dir, patch_name)
        patch = np.load(patch_item_path)
        patch = patch.astype(np.float32)
        mean, std = np.mean(patch), np.std(patch)
        min, max = np.percentile(patch, 0.5), np.percentile(patch, 99.5)
        patch = np.clip(patch, min, max)
        patch = (patch - mean) / (std + 1e-9)
        label_name = self.label_list[item]
        label_item_path = os.path.join(self.label_dir, label_name)
        label = np.loadtxt(label_item_path)
        return patch, label
    def __len__(self):
        return len(self.label_list)











