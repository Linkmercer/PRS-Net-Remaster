import os
import argparse
import sys
import numpy as np
import open3d as o3d
import random
import copy

def parse_arg():
    parser = argparse.ArgumentParser(description='data_preprocess')
    parser.add_argument('--input_path', type=str, default='./shapenet/')
    parser.add_argument('--output_path', type=str, default='../datasets/shapenet/')
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--voxel_size', type=int, default=32)
    parser.add_argument('--num_variants',  type=int, default=3)
    args = parser.parse_args()
    return args

# 忽略INFO和DEBUG日志
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)

PNG_MAGIC = b'\x89P'

def _fix_mtl_textures(obj_path: str) -> None:
    """ 解决open3d在读取JPG文件时可能会遇到数据集中为PNG文件 """
    obj_dir = os.path.dirname(obj_path)
    # 从 obj 中提取可能的 mtl 文件名
    mtl_files = []
    with open(obj_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.lower().startswith('mtllib'):
                mtl_files.extend(line.split()[1:])

    for mtl_name in mtl_files:
        mtl_path = os.path.join(obj_dir, mtl_name)
        if not os.path.exists(mtl_path):
            continue

        with open(mtl_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        modified = False
        for idx, line in enumerate(lines):
            if line.lower().startswith('map_'):
                parts = line.split(maxsplit=1)
                if len(parts) < 2:
                    continue
                tex_name = parts[1].strip()
                tex_path = os.path.join(obj_dir, tex_name)
                if not os.path.exists(tex_path):
                    continue

                # 判断真实文件是否是 PNG
                try:
                    with open(tex_path, 'rb') as tf:
                        magic = tf.read(2)
                except OSError:
                    continue

                if magic == PNG_MAGIC and tex_name.lower().endswith(('.jpg', '.jpeg')):
                    base, _ = os.path.splitext(tex_name)
                    new_tex_name = base + '.png'
                    new_tex_path = os.path.join(obj_dir, new_tex_name)
                    # 若目标名已存在则跳过重命名，直接用现有文件
                    if not os.path.exists(new_tex_path):
                        os.rename(tex_path, new_tex_path)
                    # 更新 mtl 行
                    lines[idx] = f"{parts[0]} {new_tex_name}\n"
                    modified = True

        if modified:
            with open(mtl_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            print(f"[MTL fix] {mtl_path}")

def process_obj_file(input_path):
    """读取并返回 mesh；自动修正贴图后缀"""
    if not input_path.lower().endswith('.obj'):
        print(f"Error: {input_path} is not an .obj file.")
        return None
    # 修正文件
    _fix_mtl_textures(input_path)
    print(f"Processing file: {input_path}")
    mesh = o3d.io.read_triangle_mesh(input_path)
    if mesh.is_empty():
        print(f"Failed to load mesh from {input_path}")
        return None
    return mesh


def get_closest_points(voxel_size, mesh):
    """ 计算网格最近点 """
    # 构建规则网格
    regular_points = np.zeros((voxel_size, voxel_size, voxel_size, 3), dtype=np.float32)
    for i in range(voxel_size):
        for j in range(voxel_size):
            for k in range(voxel_size):
                regular_points[i][j][k] = [i, j, k]
    # 归一化坐标
    regular_points = (regular_points+0.5)/voxel_size - 0.5
    # 创建射线投射场景
    scene = o3d.t.geometry.RaycastingScene()

    # 将mesh的顶点和三角形数据提取为Tensor类型
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    vertices_tensor = o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32)
    triangles_tensor = o3d.core.Tensor(triangles, dtype=o3d.core.Dtype.UInt32)

    # 使用Tensor数据添加到RaycastingScene
    scene.add_triangles(vertices_tensor, triangles_tensor)

    # 使用射线投射计算最近点
    closest_points = scene.compute_closest_points(regular_points)['points']
    return closest_points  # (voxel_size³, 3)


def polygon2voxel(mesh, num_samples, voxel_size):
    """ 将网格体素化 """
    # 对网格进行旋转操作
    angle = np.random.rand(3) * np.pi
    mesh.rotate(mesh.get_rotation_matrix_from_xyz(angle), center=(0, 0, 0))

    # 采样旋转后的网格
    sample_points = mesh.sample_points_uniformly(number_of_points=num_samples)

    # 将点云转换为 numpy 数组
    sample_arr = np.asarray(sample_points.points)

    # 体素化
    volume = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=1.0 / voxel_size)
    volume_arr = np.stack([vl.grid_index for vl in volume.get_voxels()])

    # 找到采样点云在每个维度上的最小值并计算计算偏移量
    coord_min = sample_arr.min(axis=0)
    offset = ((coord_min + 0.5) * voxel_size).astype(np.int32)

    # 平移体素
    volume_arr_aligned = volume_arr.copy()
    volume_arr_aligned[:, 0] += offset[0]
    volume_arr_aligned[:, 1] += offset[1]
    volume_arr_aligned[:, 2] += offset[2]

    # 创建体素模型
    model = np.zeros((voxel_size, voxel_size, voxel_size))

    for v in volume_arr_aligned:
        v[0] = min(v[0], model.shape[0] - 1)
        v[1] = min(v[1], model.shape[1] - 1)
        v[2] = min(v[2], model.shape[2] - 1)
        model[v[0], v[1], v[2]] = 1

    # 计算最近点
    closest_points = get_closest_points(voxel_size, mesh)

    return model, sample_arr, closest_points


def process_model(input_path, category, model_id, voxel_size, num_samples, output_path, num_variants, is_train=True):
    """ 处理模型并将其保存为npz文件 """
    model_file = os.path.join(input_path, category, model_id, 'models', 'model_normalized.obj')
    model_file = os.path.normpath(model_file)
    # 加载模型文件
    mesh_template = process_obj_file(model_file)
    if mesh_template is None:
        return

    for idx in range(num_variants):
        # 深拷贝原始网格，避免上一次旋转影响后续
        mesh = copy.deepcopy(mesh_template)

        # 体素化
        model_voxel, sample_points, closest_points = polygon2voxel(mesh, num_samples, voxel_size)

        # 将Tensor转numpy
        closest_points_np = closest_points.numpy()

        # 构造输出文件名，给不同 variant 加后缀 idx
        split = 'train' if is_train else 'test'
        save_dir = os.path.join(output_path, split)
        out_name = f"{model_id}_{idx}.npz"
        tsdf_file = os.path.join(save_dir, out_name)

        # 保存
        np.savez(tsdf_file,
                 model=model_voxel,
                 sample_points=sample_points,
                 closest_points=closest_points_np)


def precompute_shape_data(input_path, output_path, voxel_size, num_samples, num_variants):
    """ 处理目录中的文件并将其分为测试机和训练集 """
    # 创建测试集目录
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'test'), exist_ok=True)

    # 逐个条目处理文件，随机分配80%为训练集，20%为测试集
    categories = [d for d in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, d))]
    for category in categories:
        # 获取该类别下的所有模型
        model_ids = [d for d in os.listdir(os.path.join(input_path, category)) if
                     os.path.isdir(os.path.join(input_path, category, d))]

        # 随机打乱模型 ID 列表
        random.shuffle(model_ids)

        # 划分训练集和测试集
        split_index = int(0.8 * len(model_ids))  # 80% 用作训练集
        train_ids = model_ids[:split_index]
        test_ids = model_ids[split_index:]

        # 处理训练集
        for train_id in train_ids:
            try:
                process_model(input_path, category, train_id, voxel_size, num_samples, output_path, num_variants, is_train=True)
            except (OSError, IOError, ValueError) as e:
                print(f"continue  {category}/{train_id}：{e}")

        # 处理测试集
        for test_id in test_ids:
            try:
                process_model(input_path, category, test_id, voxel_size, num_samples, output_path, num_variants, is_train=False)
            except (OSError, IOError, ValueError) as e:
                print(f"continue {category}/{test_id}：{e}")


if __name__ == '__main__':
    for arg in sys.argv:
        print(arg)
    args = parse_arg()
    input_path = args.input_path
    output_path = args.output_path
    num_samples = args.num_samples
    voxel_size = args.voxel_size
    num_variants = args.num_variants
    precompute_shape_data(input_path, output_path, voxel_size, num_samples, num_variants)