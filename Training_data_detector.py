import SimpleITK as sitk
import numpy as np
import math
from utils import *

if __name__ == "__main__":
    path_name = '/home/zyy/training/dataset'

    for i in range(8):
        path_mhd = path_name + '0' + str(i) + '/image' + '0' + str(i) + '.mhd'
        img = sitk.ReadImage(path_mhd)
        img_arr = sitk.GetArrayFromImage(img)
        for j in range(4):
            vox_radi_path = path_name + '0' + str(i) + '/vessel' + str(j) + '/vox_radi' + '.npy'
            vox_radi_array = np.load(vox_radi_path)  # z, y, x, r
            for k in range(vox_radi_array.shape[0]):
                patch = clip_patch(img_arr, vox_radi_array[k], gap=9)  # z, y, x
                patch_save_path = path_name + '0' + str(i) + '/vessel' + str(j) + '/patch' + '/patch' + str(k) + '.npy'
                np.save(r'C:\Users\zyy\Desktop\training\dataset07\vessel3\patch\patch{}.npy'.format(i), patch)
                ber_label = ber_ref(vox_radi_array, k)
                lab_save_path = path_name + '0' + str(i) + '/vessel' + str(j) + '/detector_label/label{}.txt'.format(k)
                np.savetxt(lab_save_path, ber_label, fmt='%.5f')
