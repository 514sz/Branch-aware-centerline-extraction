import math
import numpy as np
import torch


def patch_norm(patch):
    mean, std = np.mean(patch), np.std(patch)
    min, max = np.percentile(patch, 0.5), np.percentile(patch, 99.5)
    patch = np.clip(patch, min, max)
    patch = (patch - mean) / (std + 1e-9)
    return patch


def clip_patch(img_arr, s, gap):
    # s: z,y,x
    patch = img_arr[s[0] - gap:s[0] + gap + 1, s[1] - gap:s[1] + gap + 1, s[2] - gap:s[2] + gap + 1]
    patch = patch_norm(patch)
    return patch


# return the min-distance from the centerline to point p, and index of the corresponding centerline point
def mindis(p, set):
    index = -1
    min_dis = np.sqrt(np.sum((p - set[0, :]) ** 2))
    for i in range(set.shape[0]):
        dis = np.sqrt(np.sum((p - set[i, :]) ** 2))
        if dis <= min_dis:
            min_dis = dis
            index = i
    return min_dis, index


def region_constrain(vox_ref, bound_thre):  # constrain the explore region of the agent
    # z,y,x
    zyx_min = np.min(vox_ref,axis=0)
    zyx_max = np.max(vox_ref,axis=0)
    z_min, y_min, x_min = zyx_min[0], zyx_min[1], zyx_min[2]
    z_max, y_max, x_max = zyx_max[0], zyx_max[1], zyx_max[2]
    z_left_bound = z_min - bound_thre
    z_right_bound = z_max + bound_thre
    y_left_bound = y_min - bound_thre
    y_right_bound = y_max + bound_thre
    x_left_bound = x_min - bound_thre
    x_right_bound = x_max + bound_thre
    bound = [z_left_bound, z_right_bound, y_left_bound, y_right_bound,x_left_bound, x_right_bound]
    return bound


def ber_ref(vox_radi_array, i):  # bifur_end_radi_label
    ber_label = np.zeros(3)
    if i==1:
        ber_label[0], ber_label[1], ber_label[2] = 1, 0, vox_radi_array[i, 3]
    elif i==(vox_radi_array.shape[0]-2):
        ber_label[0], ber_label[1], ber_label[2] = 0, 1, vox_radi_array[i, 3]
    else:
        ber_label[0], ber_label[1], ber_label[2] = 0, 0, vox_radi_array[i, 3]
    return ber_label


def uniform_hemisphere_samples(n):
    """
    生成半球面上均匀分布的采样点
    参数n -- 采样点数量
    返回points -- 形状为(n, 3)的numpy数组，包含归一化后的三维坐标
    """
    # 生成均匀分布的方位角（0到2π）
    phi = np.random.uniform(0, 2 * np.pi, n)

    # 生成极角（0到π/2），使用arccos(1 - u)变换保证均匀分布
    u = np.random.uniform(0, 1, n)
    theta = np.arccos(1 - u)  # 转换为极角（0到π/2）

    # 转换为笛卡尔坐标
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    # 组合成(n, 3)数组
    points = np.column_stack((x, y, z))

    # 归一化（理论上已经归一化，但保留此步骤确保数值精度）
    points /= np.linalg.norm(points, axis=1, keepdims=True)

    return points


def dot_product(vector1, vector2):
    return sum(x * y for x, y in zip(vector1, vector2))

def vector_length(vector):
    return math.sqrt(sum(x ** 2 for x in vector))



def angle_between_vectors(vector1, vector2):
    dot = dot_product(vector1, vector2)
    length1 = vector_length(vector1)
    length2 = vector_length(vector2)
    cos_angle = dot / (length1 * length2)
    if cos_angle>1:
        cos_angle=1
    if cos_angle<-1:
        cos_angle=-1
    angle = math.degrees(math.acos(cos_angle))
    return angle


def select_forward_direction(direc1, direc2, previous_direc):
    angle1, angle2 = angle_between_vectors(direc1, previous_direc), angle_between_vectors(direc2,
                                                                                          previous_direc)
    if angle1 < angle2:
        return direc1
    else:
        return direc2

def create_actions(n_actions, dis, spacing):
    offsets = []
    offset = 2.0 / n_actions
    increment = math.pi * (3.0 - math.sqrt(5.0))

    for i in range(n_actions):
        z = ((i * offset) - 1.0) + (offset / 2.0)
        r = math.sqrt(1.0 - pow(z, 2.0))
        phi = ((i + 1) % n_actions) * increment
        y = math.sin(phi) * r
        x = math.cos(phi) * r
        z = round(dis*z/spacing[2])
        y = round(dis*y/spacing[1])
        x = round(dis*x/spacing[0])
        offsets.append([z, y, x])
    offsets = np.array(offsets)
    return offsets


def normalized_entropy(probabilities):

    n = len(probabilities)

    # 计算熵
    entropy = 0.0
    for p in probabilities:
        if p>0:
            entropy -= p * math.log2(p)

    # 计算最大熵并归一化
    max_entropy = math.log2(n)
    normalized_ent = entropy / max_entropy

    return normalized_ent


def rotation_matrix(v1, v2):
    """
    计算半球轴向和CT图像Z轴间的旋转矩阵
    """

    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)

    # 归一化向量
    u1 = v1 / np.linalg.norm(v1)
    u2 = v2 / np.linalg.norm(v2)

    # 计算点积
    dot = np.dot(u1, u2)

    # 点积接近1，表示两个单位向量同方向，处理浮点误差
    if dot > 1.0 - 1e-6:
        return np.eye(3)  # 同向时返回单位矩阵
    # 点积接近-1，表示两个单位向量相反方向
    if dot < -1.0 + 1e-6:
        # 处理反向情况：构造绕垂直轴旋转180度的矩阵
        if np.allclose(u1, [1, 0, 0]) or np.allclose(u1, [-1, 0, 0]):
            # 如果u1在z轴，使用y轴作为备选
            n = np.cross(u1, [0, 1, 0])
        else:
            n = np.cross(u1, [1, 0, 0])
        n = n / np.linalg.norm(n)
        return 2 * np.outer(n, n) - np.eye(3)

    # 计算旋转轴和角度
    n = np.cross(u1, u2)
    n /= np.linalg.norm(n)
    theta = np.arccos(dot)

    # 罗德里格斯旋转公式
    K = np.array([
        [0, -n[2], n[1]],
        [n[2], 0, -n[0]],
        [-n[1], n[0], 0]
    ])

    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    return R  # 严格遵照np.dot(R, v1)或者np.dot(v2,R)的顺序，才能得到v2或者v1向量



def ray_grouping(rays_inside, origin):  # rays still inside the artery
    """
    将终点仍在冠脉血管内部的射线进行分组；不同分组表示不同血管分支
    30°和20°是需要调整的超参数
    """

    S = []  # 种子射线集合
    S.append(rays_inside[0])  # Store seed rays of different branches
    rays_inside = np.delete(rays_inside, 0, axis=0)
    rays_rows_delete1 = []
    for j in range(rays_inside.shape[0]):
        r_j_offset = rays_inside[j] - origin
        L_J = 1  # L_J表示射线j是否在不同的分支内
        for m in range(len(S)):
            s_m_offset = S[m] - origin
            angle_jm = angle_between_vectors(r_j_offset, s_m_offset)
            if angle_jm <=30:
                L_J = 0
                break
        if L_J:
            rays_rows_delete1.append(j)
            S.append(rays_inside[j])
    rays_inside = np.delete(rays_inside, rays_rows_delete1, axis=0)  # 剩余未分组射线

    # 不同分组
    groups = []
    for m in range(len(S)):
        groups.append(S[m])

    rays_rows_delete2 = []

    for j in range(rays_inside.shape[0]):
        r_j_offset = rays_inside[j] - origin
        for m in range(len(S)):
            s_m_offset = S[m] - origin
            angle_jm = angle_between_vectors(r_j_offset, s_m_offset)
            if angle_jm <20:
                groups[m] = np.vstack((groups[m], rays_inside[j]))

                rays_rows_delete2.append(j)
                break
    rays_inside = np.delete(rays_inside, rays_rows_delete2, axis=0)

    return groups

def detect_bsp(hemisphere_axis, CT_z, origin, CT_array, voxel_threshold = 2600): #
    rotatation_matrix = rotation_matrix(CT_z, hemisphere_axis)
    sample_points = uniform_hemisphere_samples(n=100)
    # y = np.dot(rotatation_matrix, hemisphere_axis)

    for i in range(sample_points.shape[0]):
        sample_points[i] = np.dot(rotatation_matrix, sample_points[i])+origin
    sample_points = np.round(sample_points).astype(int)

    original_samples_len = sample_points.shape[0]
    rows_delete = []
    for i in range(original_samples_len):  # 删除终点在血管外的射线
        z,y,x = sample_points[i]
        if CT_array[z,y,x]< voxel_threshold:  # voxel_threshold是图像预处理后的冠脉血管像素阈值
            rows_delete.append(i)
    sample_points = np.delete(sample_points, rows_delete, axis=0)

    groups = ray_grouping(sample_points, origin)
    for i in range(len(groups)):
        groups[i] = np.mean(groups[i], axis=0)

    return groups







