import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import linear_sum_assignment


# ---------------------------------- 区域划分函数
def simple_divide_cuboid_2D(xl, yl, xr, yr, n, agent_positions):
    """
    level1: 沿着短轴将长方体区域均匀划分为子区域，并分配子区域给无人机
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

def force_recursion_divide_cuboid_2D(xl, yl, xr, yr, agent_positions):
    """
    level2: 按照无人机位置划分
    递归划分区域，返回每个无人机对应的子区域，且子区域强制包含无人机位置
    """
    if len(agent_positions) == 1:
        return [(agent_positions[0], (xl, yl, xr, yr))]
    
    width = xr - xl
    height = yr - yl
    k = len(agent_positions)
    
    # 选择分割方向：宽高比或方差
    if width > height:
        direction = 'vertical'
        sorted_points = sorted(agent_positions, key=lambda p: p[0])
    else:
        direction = 'horizontal'
        sorted_points = sorted(agent_positions, key=lambda p: p[1])
    
    # 尝试找到合适的分割点
    for m in range(1, k):
        if direction == 'vertical':
            s_x = xl + width * m / k
            # 检查是否可以将前m个点放入左区域
            if m <= len(sorted_points) and (m == 0 or sorted_points[m-1][0] <= s_x) and (m == k or sorted_points[m][0] >= s_x):
                left_points = sorted_points[:m]
                right_points = sorted_points[m:]
                # 递归分割
                left = force_recursion_divide_cuboid_2D(xl, yl, s_x, yr, left_points)
                right = force_recursion_divide_cuboid_2D(s_x, yl, xr, yr, right_points)
                return left + right
        else:
            s_y = yl + height * m / k
            if m <= len(sorted_points) and (m == 0 or sorted_points[m-1][1] <= s_y) and (m == k or sorted_points[m][1] >= s_y):
                lower_points = sorted_points[:m]
                upper_points = sorted_points[m:]
                lower = force_recursion_divide_cuboid_2D(xl, yl, xr, s_y, lower_points)
                upper = force_recursion_divide_cuboid_2D(xl, s_y, xr, yr, upper_points)
                return lower + upper
    
    # 若无法找到，强制分割为两部分
    m = k // 2
    if direction == 'vertical':
        s_x = sorted_points[m][0] if m < k else xr
        left = force_recursion_divide_cuboid_2D(xl, yl, s_x, yr, sorted_points[:m])
        right = force_recursion_divide_cuboid_2D(s_x, yl, xr, yr, sorted_points[m:])
    else:
        s_y = sorted_points[m][1] if m < k else yr
        lower = force_recursion_divide_cuboid_2D(xl, yl, xr, s_y, sorted_points[:m])
        upper = force_recursion_divide_cuboid_2D(xl, s_y, xr, yr, sorted_points[m:])
    return left + right if direction == 'vertical' else lower + upper

def recursion_divide_cuboid_2D(xl, yl, xr, yr, points):
    """
    level3: 递归划分区域，返回每个无人机对应的子区域
    目标：子区域面积近似均等，不强制无人机点位于子区域内
    """
    if len(points) == 1:
        return [(points[0], (xl, yl, xr, yr))]
    
    width = xr - xl
    height = yr - yl
    k = len(points)
    
    # Options1: 选择分割方向：宽高比
    # if width < height:
    #     direction = 'vertical'
    #     sorted_points = sorted(points, key=lambda p: p[0])  # 按x坐标排序
    # else:
    #     direction = 'horizontal'
    #     sorted_points = sorted(points, key=lambda p: p[1])  # 按y坐标排序
    
    # Options2: 根据无人机分布选择分割方向
    # 计算无人机位置的x和y坐标方差
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    var_x = np.var(x_coords) if len(x_coords) > 1 else 0  # 避免单点方差为NaN
    var_y = np.var(y_coords) if len(y_coords) > 1 else 0
    
    # 选择方差更大的方向作为分割方向
    direction = 'vertical' if var_x > var_y else 'horizontal'
    if direction == 'vertical':
        sorted_points = sorted(points, key=lambda p: p[0])
    else:
        sorted_points = sorted(points, key=lambda p: p[1])
    
    # 计算分割点
    m = k // 2  # 将区域分为两部分，每部分包含 m 和 k-m 个点
    if direction == 'vertical':
        s_x = xl + width * m / k  # 按面积比例分割
        left_points = sorted_points[:m]
        right_points = sorted_points[m:]
        # 递归分割左区域和右区域
        left = recursion_divide_cuboid_2D(xl, yl, s_x, yr, left_points)
        right = recursion_divide_cuboid_2D(s_x, yl, xr, yr, right_points)
        return left + right
    else:
        s_y = yl + height * m / k  # 按面积比例分割
        lower_points = sorted_points[:m]
        upper_points = sorted_points[m:]
        # 递归分割下区域和上区域
        lower = recursion_divide_cuboid_2D(xl, yl, xr, s_y, lower_points)
        upper = recursion_divide_cuboid_2D(xl, s_y, xr, yr, upper_points)
        return lower + upper

def simple_divide_cuboid_3D(xl, yl, zl, xr, yr, zr, n, agent_positions):
    """
    level1: 沿着短轴划分
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

def force_recursion_divide_cuboid_3D(xl, yl, zl, xr, yr, zr, agent_positions):
    """
    level2: 按照无人机位置划分
    递归划分区域，返回每个无人机对应的子区域，且子区域强制包含无人机位置
    """
    if len(agent_positions) == 1:
        return [(agent_positions[0], (xl, yl, zl, xr, yr, zr))]
    
    width = xr - xl
    height = yr - yl
    depth = zr - zl
    k = len(agent_positions)
    
    # 选择分割方向：宽高比或方差，选择最长的边作为分割方向
    max_length = max(width, height, depth)
    if max_length == width:
        direction = 'x'
        sorted_points = sorted(agent_positions, key=lambda p: p[0])
    elif max_length == height:
        direction = 'y'
        sorted_points = sorted(agent_positions, key=lambda p: p[1])
    else:
        direction = 'z'
        sorted_points = sorted(agent_positions, key=lambda p: p[2])
    
    # 尝试找到合适的分割点
    for m in range(1, k):
        if direction == 'x':
            s_x = xl + width * m / k
            # 检查是否可以将前m个点放入左区域
            if m <= len(sorted_points) and (m == 0 or sorted_points[m-1][0] <= s_x) and (m == k or sorted_points[m][0] >= s_x):
                left_points = sorted_points[:m]
                right_points = sorted_points[m:]
                # 递归分割
                left = force_recursion_divide_cuboid_3D(xl, yl, zl, s_x, yr, zr, left_points)
                right = force_recursion_divide_cuboid_3D(s_x, yl, zl, xr, yr, zr, right_points)
                return left + right
        elif direction == 'y':
            s_y = yl + height * m / k
            if m <= len(sorted_points) and (m == 0 or sorted_points[m-1][1] <= s_y) and (m == k or sorted_points[m][1] >= s_y):
                lower_points = sorted_points[:m]
                upper_points = sorted_points[m:]
                lower = force_recursion_divide_cuboid_3D(xl, yl, zl, xr, s_y, zr, lower_points)
                upper = force_recursion_divide_cuboid_3D(xl, s_y, zl, xr, yr, zr, upper_points)
                return lower + upper
        else:
            s_z = zl + depth * m / k
            if m <= len(sorted_points) and (m == 0 or sorted_points[m-1][2] <= s_z) and (m == k or sorted_points[m][2] >= s_z):
                front_points = sorted_points[:m]
                back_points = sorted_points[m:]
                front = force_recursion_divide_cuboid_3D(xl, yl, zl, xr, yr, s_z, front_points)
                back = force_recursion_divide_cuboid_3D(xl, yl, s_z, xr, yr, zr, back_points)
                return front + back
    
    # 若无法找到，强制分割为两部分
    m = k // 2
    if direction == 'x':
        s_x = sorted_points[m][0] if m < k else xr
        left = force_recursion_divide_cuboid_3D(xl, yl, zl, s_x, yr, zr, sorted_points[:m])
        right = force_recursion_divide_cuboid_3D(s_x, yl, zl, xr, yr, zr, sorted_points[m:])
        return left + right
    elif direction == 'y':
        s_y = sorted_points[m][1] if m < k else yr
        lower = force_recursion_divide_cuboid_3D(xl, yl, zl, xr, s_y, zr, sorted_points[:m])
        upper = force_recursion_divide_cuboid_3D(xl, s_y, zl, xr, yr, zr, sorted_points[m:])
        return lower + upper
    else:
        s_z = sorted_points[m][2] if m < k else zr
        front = force_recursion_divide_cuboid_3D(xl, yl, zl, xr, yr, s_z, sorted_points[:m])
        back = force_recursion_divide_cuboid_3D(xl, yl, s_z, xr, yr, zr, sorted_points[m:])
        return front + back   

def recursion_divide_cuboid_3D(xl, yl, zl, xr, yr, zr, points):
    """
    level3: 递归划分区域，返回每个无人机对应的子区域
    目标：子区域体积近似均等，不强制无人机点位于子区域内
    """
    if len(points) == 1:
        return [(points[0], (xl, yl, zl, xr, yr, zr))]
    
    width = xr - xl
    height = yr - yl
    depth = zr - zl
    k = len(points)
    
    # 计算无人机位置的x、y和z坐标方差
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    z_coords = [p[2] for p in points]
    var_x = np.var(x_coords) if len(x_coords) > 1 else 0  # 避免单点方差为NaN
    var_y = np.var(y_coords) if len(y_coords) > 1 else 0
    var_z = np.var(z_coords) if len(z_coords) > 1 else 0
    
    # 选择方差更大的方向作为分割方向
    max_var = max(var_x, var_y, var_z)
    if max_var == var_x:
        direction = 'x'
        sorted_points = sorted(points, key=lambda p: p[0])
    elif max_var == var_y:
        direction = 'y'
        sorted_points = sorted(points, key=lambda p: p[1])
    else:
        direction = 'z'
        sorted_points = sorted(points, key=lambda p: p[2])
    
    # 计算分割点
    m = k // 2  # 将区域分为两部分，每部分包含 m 和 k - m 个点
    if direction == 'x':
        s_x = xl + width * m / k  # 按体积比例分割
        left_points = sorted_points[:m]
        right_points = sorted_points[m:]
        # 递归分割左区域和右区域
        left = recursion_divide_cuboid_3D(xl, yl, zl, s_x, yr, zr, left_points)
        right = recursion_divide_cuboid_3D(s_x, yl, zl, xr, yr, zr, right_points)
        return left + right
    elif direction == 'y':
        s_y = yl + height * m / k  # 按体积比例分割
        lower_points = sorted_points[:m]
        upper_points = sorted_points[m:]
        # 递归分割下区域和上区域
        lower = recursion_divide_cuboid_3D(xl, yl, zl, xr, s_y, zr, lower_points)
        upper = recursion_divide_cuboid_3D(xl, s_y, zl, xr, yr, zr, upper_points)
        return lower + upper
    else:
        s_z = zl + depth * m / k  # 按体积比例分割
        front_points = sorted_points[:m]
        back_points = sorted_points[m:]
        # 递归分割前区域和后区域
        front = recursion_divide_cuboid_3D(xl, yl, zl, xr, yr, s_z, front_points)
        back = recursion_divide_cuboid_3D(xl, yl, s_z, xr, yr, zr, back_points)
        return front + back

# ---------------------------------- 通用函数
def assign_drones_to_targets_2d(agent_positions, target_positions):
    """
    二维n对n分配函数
    """
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
    """
    三维n对n分配函数
    """
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

def plot_cuboid_2D(xl, yl, xr, yr, subregions, assignments):
    """
    可视化长方体区域、子区域、无人机与对应分配区域
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.tab20.colors  # 使用不同颜色区分子区域
    all_x = []
    all_y = []

    # 绘制矩形
    rect = Rectangle((xl, yl), xr - xl, yr - yl, 
                     edgecolor='b', facecolor='none')
    ax.add_patch(rect)
    # 绘制子区域
    for i, region in enumerate(subregions):
        sub_rect = Rectangle((region[0][0], region[0][1]), 
                             region[1][0] - region[0][0], 
                             region[1][1] - region[0][1],
                             linewidth=2, edgecolor='r', 
                             facecolor=colors[i % len(colors)], alpha=0.5)
        ax.add_patch(sub_rect)
        all_x.extend([region[0][0], region[1][0]])
        all_y.extend([region[0][1], region[1][1]])
    
    # 绘制无人机位置和分配的子区域
    for assignment in assignments:
        drone_pos, center = assignment
        ax.scatter(drone_pos[0], drone_pos[1], c='r', marker='^', s=100, label='Drone Position')
        ax.scatter(center[0], center[1], c='b', marker='^', s=100, label='Region Center Position')
        ax.plot([drone_pos[0], center[0]], [drone_pos[1], center[1]], 'k--')
    
    if all_x and all_y:
        ax.set_xlim(min(all_x)-1.0, max(all_x)+1.0)
        ax.set_ylim(min(all_y)-1.0, max(all_y)+1.0)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    plt.title('2D Region Decomposition and Assignments')
    plt.show()

def plot_cuboid_3D(xl, yl, zl, xr, yr, zr, subregions, assignments):
    """
    可视化长方体区域、子区域、无人机与对应分配区域
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.tab20.colors  # 使用不同颜色区分子区域
    all_x = []
    all_y = []
    all_z = []

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
    for i, region in enumerate(subregions):
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
        ax.add_collection3d(Poly3DCollection(faces, facecolor=colors[i % len(colors)], linewidths=2, edgecolors='r', alpha=0.4))
        all_x.extend([region[0][0], region[1][0]])
        all_y.extend([region[0][1], region[1][1]])
        all_z.extend([region[0][2], region[1][2]])


    # 绘制无人机位置和分配的子区域
    for assignment in assignments:
        drone_pos, center = assignment
        ax.scatter(drone_pos[0], drone_pos[1], drone_pos[2], c='r', marker='^', s=100, label='Drone Position')
        ax.scatter(center[0], center[1], center[2], c='b', marker='^', s=100, label='Region Center Position')
        ax.plot([drone_pos[0], center[0]], [drone_pos[1], center[1]], [drone_pos[2], center[2]], 'k--')

    if all_x and all_y and all_z:
        ax.set_xlim(min(all_x) - 1.0, max(all_x) + 1.0)
        ax.set_ylim(min(all_y) - 1.0, max(all_y) + 1.0)
        ax.set_zlim(min(all_z) - 1.0, max(all_z) + 1.0)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('3D Region Decomposition and Assignments')
    plt.show()

 

# ---------------------------------- main
# parameters
cuboid_ll = [0, 0, 0]
cuboid_tr = [20, 20, 10]
n_a = 5
agent_positions = [
    (1, 1, 3),
    (5, 1, 1),
    (1, 6, 5),
    (14, 14, 8),
    (19, 19, 9)
]
agent_positions_2d = [(x, y) for x, y, _ in agent_positions]

# algorithm
# test 2D
# ---- method1
subregions, assignments = simple_divide_cuboid_2D(cuboid_ll[0], cuboid_ll[1], cuboid_tr[0], cuboid_tr[1],
                                                  n_a, agent_positions_2d)
plot_cuboid_2D(cuboid_ll[0], cuboid_ll[1], cuboid_tr[0], cuboid_tr[1],
               subregions, assignments)
# ---- method2
regions = force_recursion_divide_cuboid_2D(cuboid_ll[0], cuboid_ll[1], cuboid_tr[0], cuboid_tr[1], 
                                           agent_positions_2d)
subregions = []
assignments = []
for i, (point, region) in enumerate(regions):
    xl, yl, xr, yr = region
    subregions.append(((xl, yl), (xr, yr)))
    assignments.append((point, ((xl + xr) / 2, (yl + yr) / 2)))
plot_cuboid_2D(cuboid_ll[0], cuboid_ll[1], cuboid_tr[0], cuboid_tr[1],
               subregions, assignments)
# ---- method3
regions = recursion_divide_cuboid_2D(cuboid_ll[0], cuboid_ll[1], cuboid_tr[0], cuboid_tr[1], 
                                     agent_positions_2d)
subregions = []
assignments = []
for i, (point, region) in enumerate(regions):
    xl, yl, xr, yr = region
    subregions.append(((xl, yl), (xr, yr)))
    assignments.append((point, ((xl + xr) / 2, (yl + yr) / 2)))
plot_cuboid_2D(cuboid_ll[0], cuboid_ll[1], cuboid_tr[0], cuboid_tr[1],
               subregions, assignments)

# test 3D
# ---- method1
subregions, assignments = simple_divide_cuboid_3D(cuboid_ll[0], cuboid_ll[1], cuboid_ll[2],
                                                  cuboid_tr[0], cuboid_tr[1], cuboid_tr[2],
                                                  n_a, agent_positions)
plot_cuboid_3D(cuboid_ll[0], cuboid_ll[1], cuboid_ll[2],
               cuboid_tr[0], cuboid_tr[1], cuboid_tr[2],
               subregions, assignments)          
# ---- method2
regions = force_recursion_divide_cuboid_3D(cuboid_ll[0], cuboid_ll[1], cuboid_ll[2],
                                           cuboid_tr[0], cuboid_tr[1], cuboid_tr[2],
                                           agent_positions)
subregions = []
assignments = []
for i, (point, region) in enumerate(regions):
    xl, yl, zl, xr, yr, zr = region
    subregions.append(((xl, yl, zl), (xr, yr, zr)))
    assignments.append((point, ((xl + xr) / 2, (yl + yr) / 2, (zl + zr) / 2)))
plot_cuboid_3D(cuboid_ll[0], cuboid_ll[1], cuboid_ll[2],
               cuboid_tr[0], cuboid_tr[1], cuboid_tr[2],
               subregions, assignments)
# ---- method3
regions = recursion_divide_cuboid_3D(cuboid_ll[0], cuboid_ll[1], cuboid_ll[2],
                                     cuboid_tr[0], cuboid_tr[1], cuboid_tr[2],
                                     agent_positions)
subregions = []
assignments = []
for i, (point, region) in enumerate(regions):
    xl, yl, zl, xr, yr, zr = region
    subregions.append(((xl, yl, zl), (xr, yr, zr)))
    assignments.append((point, ((xl + xr) / 2, (yl + yr) / 2, (zl + zr) / 2)))
plot_cuboid_3D(cuboid_ll[0], cuboid_ll[1], cuboid_ll[2],
               cuboid_tr[0], cuboid_tr[1], cuboid_tr[2],
               subregions, assignments)
