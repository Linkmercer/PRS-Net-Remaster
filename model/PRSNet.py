import torch
import torch.nn as nn
import torch.nn.functional as F

def normalize(x, enddim=4):
    """ L2归一化 """
    x = x/(1E-12 + torch.norm(x[:,:enddim], dim=1, p=2, keepdim=True))
    return x

class PRSNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=4, kernel_size=3, pool_size=2, num_plane=3, num_quat=3, linear_num=3, grid_size=32,  conv_num=5, plane_bias=None, quat_bias=None):
        super(PRSNet, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.num_plane = num_plane
        self.num_quat = num_quat
        self.linear_num = linear_num
        self.grid_size = grid_size
        self.conv_num = conv_num

        if plane_bias is None:
            plane_bias = [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0]
            ]
        if quat_bias is None:
            quat_bias = [
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]

        layers = []
        in_cn = input_channels
        out_cn = output_channels
        for _ in range(conv_num):
            # 卷积层
            layers.append(nn.Conv3d(in_cn, out_cn, kernel_size=kernel_size, stride=1, padding=1))
            # 最大池化层
            layers.append(nn.MaxPool3d(kernel_size=pool_size))
            # 激活函数
            layers.append(nn.LeakyReLU(0.2, True))
            in_cn = out_cn
            out_cn = out_cn * 2
            grid_size = grid_size // pool_size
        out_cn //= 2

        layers.append(torch.nn.Flatten())
        # 将卷积、池化和激活函数合并
        self.conv_layers = nn.Sequential(*layers)

        flat_dim = out_cn * grid_size * grid_size * grid_size
        self.activation = nn.LeakyReLU(0.2, inplace=True)

        # 全连接层，全连接层，预测平面
        self.plane_heads = nn.ModuleList()
        for i in range(num_plane):
            layers = [nn.Linear(flat_dim, flat_dim // 2), self.activation]
            layers += [nn.Linear(flat_dim // 2, flat_dim // 4), self.activation]
            last = nn.Linear(flat_dim // 4, 4)
            last.bias.data = torch.tensor(plane_bias[i], dtype=torch.float)
            layers.append(last)
            self.plane_heads.append(nn.Sequential(*layers))

        # 全连接层，预测旋转轴
        self.quat_heads = nn.ModuleList()
        for i in range(num_quat):
            layers = [nn.Linear(flat_dim, flat_dim // 2), self.activation]
            layers += [nn.Linear(flat_dim // 2, flat_dim // 4), self.activation]
            last = nn.Linear(flat_dim // 4, 4)
            last.bias.data = torch.tensor(quat_bias[i], dtype=torch.float)
            layers.append(last)
            self.quat_heads.append(nn.Sequential(*layers))

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)

        planes = [normalize(layer(x), 3) for layer in self.plane_heads]
        quats = [normalize(layer(x), 4) for layer in self.quat_heads]

        return torch.stack(planes, dim=1), torch.stack(quats, dim=1)


def reflect_points_on_plane(points, planes, eps=1e-12):
    """ 计算反射平面对称点 """
    batch_size = points.shape[1]
    num_planes = planes.shape[1]

    # 拆出法向量 n 和偏置 d
    ns = planes[..., :3]                      # [b,p,3]
    ds = planes[...,  3]                      # [b,p]

    # 将法向量归一化
    ns_norm = ns.norm(dim=2, keepdim=True)        # [b,p,1]
    n_unit = ns / (ns_norm + eps)                # [b,p,3]

    # 将偏置归一化（防止除0）
    d_unit = ds.unsqueeze(-1) / (ns_norm + eps)   # [b,p,1]

    # 把点，法向量和偏置扩维
    pts_e = points.unsqueeze(1).expand(-1, num_planes, -1, -1)     # [b,p,n,3]
    n_e = n_unit.unsqueeze(2).expand(-1, -1,  batch_size, -1)     # [b,p,n,3]
    d_e = d_unit.unsqueeze(2).expand(-1, -1,  batch_size, -1)     # [b,p,n,1]

    # 计算反射平面对称点，使用单位化的n
    inner = (pts_e * n_e).sum(dim=-1, keepdim=True) + d_e  # [b,p,n,1]
    reflected = pts_e - 2 * inner * n_e                       # [b,p,n,3]

    return reflected


def sym_points_on_quat(points, quads):
    """ 计算关于旋转轴的对称点 """
    batch_size = points.shape[1]
    num_quads = quads.shape[1]

    # 归一化预测四元组
    q = F.normalize(quads, p=2, dim=-1)                  # [b,q,4]

    # 旋转轴维度扩展，计算共轭
    q_e = q.unsqueeze(2).expand(-1, -1, batch_size, -1)        # [b,q,n,4]
    q_conj = q_e.clone(); q_conj[..., 1:] *= -1          # 共轭四元组

    # 把点移动到以质心为原点的坐标系，用点减中心
    pts_e = points.unsqueeze(1).expand(-1, num_quads, -1, -1)  # [b,q,n,3]
    center = points.mean(dim=1, keepdim=True)          # [b,1,3]
    ctr_e = center.unsqueeze(2).expand(-1, num_quads, batch_size, -1)   # [b,q,n,3]
    pts_c = pts_e - ctr_e                             # 中心化

    # 构造纯虚四元组 v4 = [0, pts_c]
    v4 = torch.cat([torch.zeros_like(pts_c[...,:1]), pts_c], dim=-1)  # [b,q,n,4]

    # Hamilton 乘法： q * v4 * q_conj
    def hprod(a, b):
        w1, x1, y1, z1 = a.unbind(-1)
        w2, x2, y2, z2 = b.unbind(-1)
        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dim=-1)

    tmp = hprod(q_e, v4)
    rot = hprod(tmp, q_conj)            # [b,q,n,4]
    point = rot[..., 1:] + ctr_e           # 提取 xyz 并还原中心

    return point


class SymPlaneLoss(nn.Module):
    """ 计算对称平面距离损失 """

    def forward(self, points, closest_points, voxel, planes):
        batch_size, grid_resolution = voxel.shape[0], voxel.shape[1]
        num_planes = planes.shape[1]

        # 反射对称点 [b,p,n,3]
        pts_ref = reflect_points_on_plane(points, planes)

        # 映射到网格中心索引(在规则网络算的最近点，又返回规则网络算最小距离)
        idx = torch.floor((pts_ref + 0.5) * grid_resolution).long()
        idx = idx[..., 0]*grid_resolution*grid_resolution + idx[..., 1]*grid_resolution + idx[..., 2]
        idx = idx.clamp(0, grid_resolution**3 - 1)  # [b,p,n]

        # 采样最近点[b,g^3,3]
        cp_flat = closest_points.view(batch_size, -1, 3)  # 索引
        cp_e = cp_flat.unsqueeze(1).expand(-1, num_planes, -1, -1)  # 扩维
        cp_sel = torch.gather(cp_e, 2, idx.unsqueeze(-1).expand(-1, -1, -1, 3))  # 取点

        # mask 无效体素，惩罚不合格对称点
        mask = 1 - torch.gather(voxel.view(batch_size, 1, -1).expand(-1, num_planes, -1), 2, idx)

        # 计算平方误差并平均（和论文有差别）
        diff = (pts_ref - cp_sel) * mask.unsqueeze(-1)
        loss = diff.pow(2).sum(dim=-1).sum(dim=-1).mean()
        return loss


class SymQuatLoss(nn.Module):
    """ 计算对称旋转距离损失 """
    def forward(self, points, closest_points, voxel, quads):
        batch_size, grid_resolution = voxel.shape[0], voxel.shape[1]
        num_quads = quads.shape[1]

        # 旋转对称点
        pts_rot = sym_points_on_quat(points, quads)

        # 映射到网格中心索引(在规则网络算的最近点，又返回规则网络算最小距离)
        idx = torch.floor((pts_rot + 0.5) * grid_resolution).long()
        idx = idx[..., 0]*grid_resolution*grid_resolution + idx[..., 1]*grid_resolution + idx[..., 2]
        idx = idx.clamp(0, grid_resolution**3 - 1)  # [b,q,n]

        # 采样最近点[b,g^3,3]
        cp_flat = closest_points.view(batch_size, -1, 3)  # 索引
        cp_e = cp_flat.unsqueeze(1).expand(-1, num_quads, -1, -1)  # 扩维
        cp_sel = torch.gather(cp_e, 2, idx.unsqueeze(-1).expand(-1, -1, -1, 3))  # 取点

        # mask 无效体素，惩罚不合格对称点
        mask = 1 - torch.gather(voxel.view(batch_size, 1, -1).expand(-1, num_quads, -1), 2, idx)

        # 计算平方误差并平均（和论文有差别）
        diff = (pts_rot - cp_sel) * mask.unsqueeze(-1)
        loss = diff.pow(2).sum(dim=-1).sum(dim=-1).mean()
        return loss


class RegLoss(nn.Module):
    """ 计算正则损失 """
    def forward(self, vectors):
        num_vectors = vectors.shape[1]

        # L2 归一化
        normalized = F.normalize(vectors, p=2, dim=2)  # [b,K,D]

        # 计算Gram矩阵
        matrix = torch.bmm(normalized, normalized.transpose(1, 2))

        # 构造单位矩阵并做差
        identity = torch.eye(num_vectors, device=matrix.device, dtype=matrix.dtype).unsqueeze(0)
        diff = matrix - identity

        # 计算 Frobenius 范数的平方并对 batch 平均
        frob2 = (diff*diff).sum(dim=(1, 2))
        return frob2.mean()


class PRSLoss(nn.Module):
    """ 计算总损失 """
    def __init__(self, wr=25.0):
        super().__init__()
        self.sp = SymPlaneLoss()
        self.sq = SymQuatLoss()
        self.rl = RegLoss()
        self.wr = wr

    def forward(self, voxel, points, closest_points, planes, quads):
        batch_size = voxel.shape[0]
        cp_flat = closest_points.view(batch_size, -1, 3)

        sym_plane = self.sp(points, cp_flat, voxel, planes)
        sym_quat = self.sq(points, cp_flat, voxel, quads)
        sym_loss = sym_plane + sym_quat

        # 忽略最后 bias d
        reg_plane = self.rl(planes[:, :, :3])
        # 忽略实部 w
        reg_quat = self.rl(quads[:, :, 1:])
        reg_loss = reg_plane + reg_quat

        return sym_loss + self.wr * reg_loss
