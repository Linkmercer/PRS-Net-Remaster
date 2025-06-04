import argparse
import torch
from torch.utils.data import DataLoader
import plotly.graph_objects as go
import numpy as np
from model.PRSNet import PRSNet
from data.load_dataset import LoadDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Testing")
    parser.add_argument('--model_path', type=str, default='trained_model/best_model/best_model.pth')
    parser.add_argument('--data_dir', type=str, default='datasets/shapenet/test', help='path to dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_planes', type=int, default=1, choices=[0, 1, 2, 3], help='number of displayed planes')
    parser.add_argument('--num_quads', type=int, default=0, choices=[0, 1, 2, 3], help='number displayed of quads')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    dataset = LoadDataset(args.data_dir, device=device)
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=False,)

    # 加载模型
    model = PRSNet().to(device)
    state = torch.load(args.model_path, map_location=device)
    if isinstance(state, dict) and 'model_state' in state:
        model.load_state_dict(state['model_state'])
    else:
        model.load_state_dict(state)
    model.eval()

    # 遍历样本，提取对称平面和旋转轴
    for i, (voxel, points, _) in enumerate(loader):
        sp = points[0].cpu().numpy()
        vox = voxel.unsqueeze(1).to(device)
        with torch.no_grad():
            planes, quads = model(vox)
        planes = planes.squeeze(2)[0].cpu().numpy()
        quads = quads.squeeze(2)[0].cpu().numpy()

        # 绘制点云散点
        scatter = go.Scatter3d(
            x=sp[:, 0], y=sp[:, 1], z=sp[:, 2],
            mode='markers',
            marker=dict(size=2, color='black'),
            name='PointCloud'
        )

        # 绘制对称平面
        surfaces = []
        grid = np.linspace(-0.6, 0.6, 20)
        xx, yy = np.meshgrid(grid, grid)
        for j, (A, B, C, D) in enumerate(planes[:args.num_planes]):
            if abs(C) > 1e-6:
                zz = (-A * xx - B * yy - D) / C
            elif abs(A) > 1e-6:
                zz = np.linspace(-0.6, 0.6, 20)
                xx = (-C * zz - B * yy - D) / A
            else:
                zz = np.linspace(-0.6, 0.6, 20)
                yy = (-A * xx - C * zz - D) / B

            surfaces.append(
                go.Surface(
                    x=xx, y=yy, z=zz,
                    showscale=False,
                    opacity=0.4,
                    name=f'Plane{j + 1}'
                )
            )

        # 绘制旋转轴
        axes = []
        for j, (_w, i0, j0, k0) in enumerate(quads[:args.num_quads]):
            v = np.array([i0, j0, k0], dtype=float)
            v /= np.linalg.norm(v)
            axes.append(
                go.Scatter3d(
                    x=[-v[0], v[0]],
                    y=[-v[1], v[1]],
                    z=[-v[2], v[2]],
                    mode='lines',
                    line=dict(width=4),
                    name=f'Axis{j + 1}'
                )
            )

        fig = go.Figure(data=[scatter, *surfaces, *axes])
        fig.update_layout(
            title=f"Sample {i} Symmetry Planes",
            scene=dict(
                xaxis=dict(range=[-0.6, 0.6]),
                yaxis=dict(range=[-0.6, 0.6]),
                zaxis=dict(range=[-0.6, 0.6]),
                aspectmode='cube'
            )
        )
        fig.show()

        cont = input('Continue (y/n)? ')
        if cont.lower().startswith('n'):
            break