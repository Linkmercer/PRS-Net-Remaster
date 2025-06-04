import torch
from torch.utils.data import Dataset, random_split, DataLoader
import numpy as np
import os

class LoadDataset(Dataset):
    def __init__(self, data_path, device="cpu", preload=True):
        """ 初始化数据集类。 """
        self.data_path = data_path
        self.device = device

        # 获取所有的 .npz 文件（每个文件对应一个模型的数据）
        self.model_files = [f for f in os.listdir(data_path) if f.endswith(".npz")]

        # 一次性加载到内存
        self.preload = preload
        self.cached = []

        if self.preload:
            # 把全部样本读进内存并转成 Tensor
            for fname in self.model_files:
                npz = np.load(os.path.join(data_path, fname))
                self.cached.append(
                    (
                        torch.tensor(npz["model"], dtype=torch.float32),
                        torch.tensor(npz["sample_points"], dtype=torch.float32),
                        torch.tensor(npz["closest_points"], dtype=torch.float32),
                    )
                )

    def __len__(self):
        """ 返回数据集的大小 """
        return len(self.model_files)

    def __getitem__(self, idx):
        """ 获取指定索引的数据项 """
        if self.preload:
            model, sample_points, closest_points = self.cached[idx]
        else:
            model_file = os.path.join(self.data_path, self.model_files[idx])
            data = np.load(model_file)
            model = torch.tensor(data["model"], dtype=torch.float32)
            sample_points = torch.tensor(data["sample_points"], dtype=torch.float32)
            closest_points = torch.tensor(data["closest_points"], dtype=torch.float32)

        return model, sample_points, closest_points