import numpy as np
# import cv2
import SimpleITK as sitk
import matplotlib.pyplot as plt

"""
physical coordinate -> voxel coordinate
"""

# CT volume spacing
def wc_vc(filename, spacing):
    reference = np.loadtxt(filename, dtype=np.float32, delimiter=' ')
    reference[:, 0] = np.round(reference[:, 0] / spacing[0], 0)  # x
    reference[:, 1] = np.round(reference[:, 1] / spacing[1], 0)  # y
    reference[:, 2] = np.round(reference[:, 2] / spacing[2], 0)  # z
    column3_reference = reference[:, 0:3]
    column3_reference = column3_reference.astype(int)  # x,y,z
    column3_reference[:, [0,2]] = column3_reference[:, [2,0]]  # z,y,x
    voxel_uniq, index = np.unique(column3_reference, return_index=True, axis=0)  # delete repeated voxel coordinate；index of reference.txt
    index2 = np.sort(index)
    index1 = np.argsort(index)
    radius = reference[index2, 3]
    radius.resize(len(radius), 1)
    voxel_uniq = voxel_uniq[index1, :]
    vox_radi_uniq = np.append(voxel_uniq, radius, axis=1)  # z,y,x,radius
    return vox_radi_uniq


if __name__ == "__main__":
    path_name = '/home/zyy/training/dataset'
    for i in range(8):
        for j in range(4):
            path_ref = path_name + '0' + str(i) + '/vessel' + str(j) + '/reference.txt'
            path_CT = path_name + '0' + str(i) + '/image' + '0' + str(i) + '.mhd'
            image = sitk.ReadImage(path_CT)
            Spacing = image.GetSpacing()
            vox_radi_ref = wc_vc(path_ref, Spacing)
            vox_radi_path = path_name + '0' + str(i) + '/vessel' + str(j) + '/vox_radi' + '.npy'
            np.save(vox_radi_path, vox_radi_ref)


