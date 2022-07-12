import numpy as np
from sklearn.preprocessing import normalize

angel_thresh = 40
# ct iamge array
ct_arr = np.load('ct_arr.npy')

# rays normalize
rays = normalize(np.array([[1,-1,-1],[1,-1,0],[1,-1,1],[1,0,-1],[1,0,0],[1,0,1],[1,1,-1],[1,1,0],[1,1,1],
                           [0,-1,-1],[0,-1,0],[0,-1,1],[0,0,-1],[0,0,1],[0,1,-1],[0,1,0],[0,1,1]]))
rays_label = np.zeros(17)
radius = 1
p1,p2 = np.array([4,7,8]),np.array([2,6,7])
# p1: last point; p2: current point; radius: radius at p2
def bran_detec(p1, p2, rays, radius):
    direc = p2-p1
    lenth = np.linalg.norm(direc)  # length normalize
    norm_direc = direc/lenth  # direc normalize
    offset = norm_direc-np.array([1,0,0])
    rays += offset
    rays *= radius*4
    rays_end = rays+p2

    set1, set2 = ray_set(rays_end)
    return np.mean(set1, axis=0), np.mean(set2, axis=0)

def cal_angle(v1,v2):
    dot_product = np.dot(v1,v2)
    arccos = np.arccos(dot_product/(np.linalg.norm(v1)*np.linalg.norm(v2)))
    angle = np.degrees(arccos)
    return angle

def ray_set(rays_end):
    set1 = np.zeros((0,3))
    set2 = np.zeros((0,3))
    for i in range(len(rays_end)):
        if rays_end[i] >= 200:
            base= rays_end[i]
            for j in range(i, len(rays_end)):
                if rays_end[j]>=200 and cal_angle(base, rays_label[j]) >= angel_thresh:
                    set2 = np.append(set2, rays_end[j], axis=0)
                elif rays_end[j]>=200 and cal_angle(base, rays_label[j]) < angel_thresh:
                    set1 = np.append(set1, rays_end[j], axis=0)
            break
    return set1, set2













