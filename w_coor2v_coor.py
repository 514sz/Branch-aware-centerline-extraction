import numpy as np
# import cv2
import SimpleITK as sitk
import matplotlib.pyplot as plt


def clip_patch(filename, vox_radi_ref):
    image = sitk.ReadImage(filename)
    img_arr = sitk.GetArrayFromImage(image)
    gap = 12
    z, x, y = int(vox_radi_ref[0, 2]), int(vox_radi_ref[0, 0]), int(vox_radi_ref[0, 1])
    patch = img_arr[z-gap:z+gap+1, y-gap:y+gap+1, x-gap:x+gap+1] # z,y,x对应img_arr的slice，行号，列号
    # scan = patch[100, ...]
    # scan2 = img_arr[271, ...]
    # scan3 = img_arr[250, ...]
    # plt.imshow(scan3, cmap='gray')
    # plt.show()
    # plt.imshow(scan, cmap='gray')
    # plt.show()
    # plt.imshow(scan2, cmap='gray')
    # plt.show()
    return patch

# path = r'C:\Users\zyy\Desktop\dataset00\image00.mhd'
# image =sitk.ReadImage(path)
# img_arr = sitk.GetArrayFromImage(image)
# gap = 12
# patch = img_arr[z-gap:z+gap+1, x-gap:x+gap+1, y-gap:y+gap+1]
# spacing = image.GetSpacing()
# slice = 271
# scan = np.squeeze(img_arr[slice, ...]) # if the image is 3d, the slice is integer
#
# plt.imshow(scan, cmap='gray')
# # plt.axis('off')
# plt.show()
# # cv2.imwrite('1.png', image)

def read_txt(filename):
    reference = np.loadtxt(filename, dtype=np.float32, delimiter=' ')
    # plt.imshow(reference[0, ...], cmap='gray')
    # data = data/np.array([0.363281011581, 0.363281011581, 0.40000000596, 1.0, 1.0])
    reference[:, 0] = np.round(reference[:, 0] / 0.363281011581, 0) # x
    reference[:, 1] = np.round(reference[:, 1] / 0.363281011581, 0) # y
    reference[:, 2] = np.round(reference[:, 2] / 0.40000000596, 0) #z
    column3_reference = reference[:, 0:3]
    voxel_uniq, index = np.unique(column3_reference, return_index=True, axis=0)  # voxel_uniq:元素从小到大排列；index:新列表元素在旧列表中的位置
    index2 = np.sort(index)
    index1 = np.argsort(index)
    radius = reference[index2, 3]
    radius.resize(len(radius), 1)
    voxel_uniq = voxel_uniq[index1, :]
    vox_radi_uniq = np.append(voxel_uniq, radius, axis=1)
    return vox_radi_uniq


if __name__ == "__main__":
    path_ref = r'dataset00/vessel0/reference.txt'
    vox_radi_ref = read_txt(path_ref)
    np.save(r'dataset00/vessel0/vox_radi.npy', vox_radi_ref)
    # path_mhd = r'dataset00/image00.mhd'
    # patch = clip_patch(path_mhd, vox_radi_ref)
    # np.save(r'dataset00/vessel0/patch0', patch)
    # print(vox_radi_ref)
