import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from model.PRSNet import PRSNet, PRSLoss
from data.load_dataset import LoadDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--data_dir', type=str, default='datasets/shapenet', help='path to dataset')
    parser.add_argument('--save_dir', type=str, default='trained_model', help='path to save models')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--resume', action='store_true', help='path to latest checkpoint')
    args = parser.parse_args()
    return args


def set_seed(seed=1):
    """ 固定随机种子，确保可复现 """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(state, save_dir, epoch):
    """ 每轮结束后存储完整训练状态 """
    fn = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(state, fn)


def load_checkpoint(model, optimizer, save_dir):
    """ 加载最新 checkpoint 并恢复模型/优化器状态 """
    files = [f for f in os.listdir(save_dir)
             if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
    if not files:
        return 1, float('inf')
    # 按 epoch 编号排序，取最新
    files = sorted(files, key=lambda f: int(f.split('_')[-1].split('.')[0]))
    ckpt = torch.load(os.path.join(save_dir, files[-1]))
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optim_state'])
    print(f"Resumed from epoch {ckpt['epoch']} "
          f"(saved val_loss={ckpt['best_val_loss']:.6f})")
    # 下一轮从 ckpt.epoch+1 开始
    return ckpt['epoch'] + 1, ckpt['best_val_loss']


if __name__ == '__main__':
    args = parse_args()
    ckpt_dir = os.path.join(args.save_dir, 'checkpoints')
    best_dir = os.path.join(args.save_dir, 'best_model')
    last_dir = os.path.join(args.save_dir, 'last_model')  # 新增目录，用于保存最后一次模型
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(last_dir, exist_ok=True)  # 创建目录

    set_seed()

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据集
    train_ds = LoadDataset(os.path.join(args.data_dir, 'train'), device=device)
    val_ds = LoadDataset(os.path.join(args.data_dir, 'test'), device=device)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = PRSNet().to(device)
    loss_fn = PRSLoss().to(device)
    # Adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 调度器，每50个epoch更新一次学习率，每次减半
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    start_epoch = 1
    best_val_loss = float('inf')
    # 如果参数中有resume则恢复上次训练状态
    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(
            model, optimizer, ckpt_dir)

    for epoch in range(start_epoch, args.epochs + 1):
        # 训练阶段
        model.train()
        train_loss_sum, train_samples = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for voxel, sample_points, closest_points in pbar:
            points = sample_points.to(device, non_blocking=True)
            voxels = voxel.to(device, non_blocking=True).unsqueeze(1)
            cp = closest_points.to(device, non_blocking=True)

            optimizer.zero_grad()
            planes, quads = model(voxels)
            loss = loss_fn(voxels.squeeze(1), points, cp, planes, quads)
            loss.backward()
            optimizer.step()

            bsize = sample_points.size(0)
            train_loss_sum += loss.item() * bsize
            train_samples += bsize
            pbar.set_postfix(train_loss=train_loss_sum / train_samples)
        scheduler.step()  # 学习率调度

        # 验证阶段
        model.eval()
        val_loss_sum, val_samples = 0.0, 0
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [  Val ]")
            for voxel, sample_points, closest_points in pbar:
                points = sample_points.to(device, non_blocking=True)
                voxels = voxel.to(device, non_blocking=True).unsqueeze(1)
                cp = closest_points.to(device, non_blocking=True)

                planes, quads = model(voxels)
                loss = loss_fn(voxels.squeeze(1), points, cp, planes, quads)

                bsize = sample_points.size(0)
                val_loss_sum += loss.item() * bsize
                val_samples += bsize
                pbar.set_postfix(val_loss=val_loss_sum / val_samples)

        avg_val_loss = val_loss_sum / max(val_samples, 1)
        print(f"Epoch {epoch:03d} → "
              f"Train: {train_loss_sum / train_samples:.6f}, "
              f"Val: {avg_val_loss:.6f}")

        # 保存最好的模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(best_dir, 'best_model.pth'))
            print(f"✅ New best model saved (val_loss={best_val_loss:.6f})")

        # 保存检查点
        save_checkpoint({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }, ckpt_dir, epoch)

    print("🏁 Training completed.")

    # 保存最后一次训练得到的模型
    torch.save(model.state_dict(), os.path.join(last_dir, 'last_model.pth'))
    print("📦 Last epoch model saved.")

    # 训练结束后重置检查点
    for f in os.listdir(ckpt_dir):
        if f.startswith('checkpoint_epoch_') and f.endswith('.pth'):
            os.remove(os.path.join(ckpt_dir, f))
