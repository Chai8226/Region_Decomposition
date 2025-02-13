import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import linear_sum_assignment


# ---------------------------------- 通用函数
## n对n分配函数
def assign_drones_to_targets_2d(agent_positions, target_positions):
    # 计算无人机组和目标组之间的欧氏距离矩阵
    num_drones = len(agent_positions)
    num_targets = len(target_positions)
    distance_matrix = np.zeros((num_drones, num_targets))
    for i in range(num_drones):
        for j in range(num_targets):
            dx = agent_positions[i][0] - target_positions[j][0]
            dy = agent_positions[i][1] - target_positions[j][1]
            distance_matrix[i][j] = np.sqrt(dx**2 + dy**2)
    
    # 使用匈牙利算法求解最小权匹配
    row_indices, col_indices = linear_sum_assignment(distance_matrix)
    
    # 生成分配结果
    assignments = []
    for i in range(len(row_indices)):
        drone_index = row_indices[i]
        target_index = col_indices[i]
        assignment = (agent_positions[drone_index], target_positions[target_index])
        assignments.append(assignment)
    return assignments

def assign_drones_to_targets_3d(agent_positions, target_positions):
    # 计算无人机组和目标组之间的欧氏距离矩阵
    num_drones = len(agent_positions)
    num_targets = len(target_positions)
    distance_matrix = np.zeros((num_drones, num_targets))
    for i in range(num_drones):
        for j in range(num_targets):
            # 考虑 3D 坐标，增加 z 轴距离的计算
            dx = agent_positions[i][0] - target_positions[j][0]
            dy = agent_positions[i][1] - target_positions[j][1]
            dz = agent_positions[i][2] - target_positions[j][2]
            # 3D 欧氏距离公式
            distance_matrix[i][j] = np.sqrt(dx**2 + dy**2 + dz**2)
    # 使用匈牙利算法求解最小权匹配
    row_indices, col_indices = linear_sum_assignment(distance_matrix)
    # 生成分配结果
    assignments = []
    for i in range(len(row_indices)):
        drone_index = row_indices[i]
        target_index = col_indices[i]
        assignment = (agent_positions[drone_index], target_positions[target_index])
        assignments.append(assignment)
    return assignments

## 可视化
def plot_cuboid_2D(xl, yl, xr, yr, subregions, assignments):
    """
    可视化长方体区域、子区域和无人机位置
    """
    fig, ax = plt.subplots()
    # 绘制矩形
    rect = Rectangle((xl, yl), xr - xl, yr - yl, edgecolor='b', facecolor='none')
    ax.add_patch(rect)
    # 绘制子区域
    for region in subregions:
        sub_rect = Rectangle((region[0][0], region[0][1]), 
                             region[1][0] - region[0][0], 
                             region[1][1] - region[0][1],
                             edgecolor='r', facecolor='cyan', alpha=0.5)
        ax.add_patch(sub_rect)
    # 绘制无人机位置和分配的子区域
    for assignment in assignments:
        drone_pos, center = assignment
        ax.scatter(drone_pos[0], drone_pos[1], c='r', marker='^', s=100, label='Drone Position')
        ax.scatter(center[0], center[1], c='b', marker='^', s=100, label='Region Center Position')
        ax.plot([drone_pos[0], center[0]], [drone_pos[1], center[1]], 'k--')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.title('2D Region Decomposition and Assignments')
    plt.axis('equal')
    plt.show()

def plot_cuboid_3D(xl, yl, zl, xr, yr, zr, subregions, assignments):
    """
    可视化长方体区域、子区域和无人机位置
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制长方体
    cuboid = np.array([
        [xl, yl, zl],
        [xr, yl, zl],
        [xr, yr, zl],
        [xl, yr, zl],
        [xl, yl, zr],
        [xr, yl, zr],
        [xr, yr, zr],
        [xl, yr, zr]
    ])
    ax.scatter(cuboid[:, 0], cuboid[:, 1], cuboid[:, 2], c='b', marker='o')

    # 绘制子区域
    for region in subregions:
        vertices = [
            [region[0][0], region[0][1], region[0][2]],
            [region[1][0], region[0][1], region[0][2]],
            [region[1][0], region[1][1], region[0][2]],
            [region[0][0], region[1][1], region[0][2]],
            [region[0][0], region[0][1], region[1][2]],
            [region[1][0], region[0][1], region[1][2]],
            [region[1][0], region[1][1], region[1][2]],
            [region[0][0], region[1][1], region[1][2]]
        ]
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]],
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[1], vertices[2], vertices[6], vertices[5]],
            [vertices[0], vertices[3], vertices[7], vertices[4]]
        ]
        ax.add_collection3d(Poly3DCollection(faces, facecolors='cyan', linewidths=1, edgecolors='r', alpha=0.1))

    # 绘制无人机位置和分配的子区域
    for assignment in assignments:
        drone_pos, center = assignment
        ax.scatter(drone_pos[0], drone_pos[1], drone_pos[2], c='r', marker='^', s=100, label='Drone Position')
        ax.scatter(center[0], center[1], center[2], c='b', marker='^', s=100, label='Region Center Position')
        ax.plot([drone_pos[0], center[0]], [drone_pos[1], center[1]], [drone_pos[2], center[2]], 'k--')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('3D Region Decomposition and Assignments')
    plt.show()


# ---------------------------------- 区域划分函数
## 2D
### level1: 沿着短轴划分
def divide_cuboid_2D(xl, yl, xr, yr, n, agent_positions):
    """
    将长方体区域均匀划分为子区域，并分配子区域给无人机
    :param xl, yl: 长方体的左下顶点坐标
    :param xr, yr: 长方体的右上顶点坐标
    :param n: 无人机数量
    :param agent_positions: 无人机的初始位置列表，格式为 [(x1, y1), (x2, y2), ...]
    :return: 子区域列表和分配结果
    """

    # 1. 计算每个维度的划分数量
    L = xr - xl
    W = yr - yl

    if L > W:
        n_x = 1
        n_y = n
    else:
        n_x = n
        n_y = 1

    delta_x = L / n_x
    delta_y = W / n_y


    # 2. 生成子区域
    subregions = []
    for i in range(n_x):
        for j in range(n_y):
            x_start = xl + i * delta_x
            x_end = xl + (i + 1) * delta_x
            y_start = yl + j * delta_y
            y_end = yl + (j + 1) * delta_y
            subregions.append(((x_start, y_start), (x_end, y_end)))

    # 3. 分配子区域给无人机
    subregions_center = []
    for region in subregions:
        subregions_center.append(
            ((region[0][0] + region[1][0]) / 2, (region[0][1] + region[1][1]) / 2)
        )
    assignments = assign_drones_to_targets_2d(agent_positions, subregions_center)
    
    return subregions, assignments

## 3D
### level1: 沿着短轴划分
def divide_cuboid_3D(xl, yl, zl, xr, yr, zr, n, agent_positions):
    """
    将长方体区域均匀划分为子区域，并分配子区域给无人机
    :param xl, yl, zl: 长方体的左下顶点坐标
    :param xr, yr, zr: 长方体的右上顶点坐标
    :param n: 无人机数量
    :param agent_positions: 无人机的初始位置列表，格式为 [(x1, y1, z1), (x2, y2, z2), ...]
    :return: 子区域列表和分配结果
    """

    # 1. 计算每个维度的划分数量
    L = xr - xl
    W = yr - yl
    H = zr - zl

    if L < W and L < H:
        n_x = n
        n_y = 1
        n_z = 1
    elif W < L and W < H:
        n_x = 1
        n_y = n
        n_z = 1
    else:
        n_x = 1
        n_y = 1
        n_z = n

    delta_x = L / n_x
    delta_y = W / n_y
    delta_z = H / n_z

    # 2. 生成子区域
    subregions = []
    for i in range(n_x):
        for j in range(n_y):
            for k in range(n_z):
                x_start = xl + i * delta_x
                x_end = xl + (i + 1) * delta_x
                y_start = yl + j * delta_y
                y_end = yl + (j + 1) * delta_y
                z_start = zl + k * delta_z
                z_end = zl + (k + 1) * delta_z
                subregions.append(((x_start, y_start, z_start), (x_end, y_end, z_end)))

    # 3. 分配子区域给无人机
    subregions_center = []
    for region in subregions:
        subregions_center.append(
            ((region[0][0] + region[1][0]) / 2, 
             (region[0][1] + region[1][1]) / 2, 
             (region[0][2] + region[1][2]) / 2)
        )
    assignments = assign_drones_to_targets_3d(agent_positions, subregions_center)

    return subregions, assignments


# ---------------------------------- main
# parameters
cuboid_ll = [0, 0, 0]
cuboid_tr = [10, 20, 30]
n_a = 3
agent_positions = [
    (1, 1, 1),
    (3, 5, 3),
    (9, 2, 5),
]
agent_positions_2d = [(x, y) for x, y, _ in agent_positions]

# algorithm
subregions, assignments = divide_cuboid_2D(cuboid_ll[0], cuboid_ll[1],
                                           cuboid_tr[0], cuboid_tr[1],
                                           n_a, agent_positions_2d)
plot_cuboid_2D(cuboid_ll[0], cuboid_ll[1], cuboid_tr[0], cuboid_tr[1],
               subregions, assignments)

subregions, assignments = divide_cuboid_3D(cuboid_ll[0], cuboid_ll[1], cuboid_ll[2],
                                           cuboid_tr[0], cuboid_tr[1], cuboid_tr[2],
                                           n_a, agent_positions)
plot_cuboid_3D(cuboid_ll[0], cuboid_ll[1], cuboid_ll[2],
               cuboid_tr[0], cuboid_tr[1], cuboid_tr[2],
               subregions, assignments)