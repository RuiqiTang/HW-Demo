"""
训练脚本
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from pathlib import Path
from datetime import datetime

from config import (
    DEVICE, DIFFUSION_CONFIG, MODEL_CONFIG, TRAIN_CONFIG,
    SPRITE_DATA_PATH, SPRITE_LABELS_PATH, WEIGHTS_DIR, TENSORBOARD_DIR
)
from dataset import get_dataloader
from models import ContextUNet
from diffusion import GaussianDiffusion


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    diffusion: GaussianDiffusion,
    optimizer: optim.Optimizer,
    epoch: int,
    device: torch.device,
    writer: SummaryWriter = None,
    use_context: bool = False
) -> float:
    """
    训练一个epoch
    
    Args:
        model: 模型
        dataloader: 数据加载器
        diffusion: 扩散模型
        optimizer: 优化器
        epoch: 当前epoch
        device: 设备
        use_context: 是否使用条件控制
        
    Returns:
        avg_loss: 平均损失
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (x, labels) in enumerate(pbar):
        x = x.to(device)
        labels = labels.to(device) if use_context else None
        
        # 随机采样时间步（归一化到[0,1]）
        t = torch.randint(
            1, diffusion.timesteps + 1,
            (x.shape[0],),
            device=device
        ).float() / diffusion.timesteps  # 归一化到[0,1]
        
        # 计算损失
        if use_context:
            loss = diffusion.train_losses(model, x, t, c=labels)
        else:
            loss = diffusion.train_losses(model, x, t)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # 更新进度条
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # 记录到tensorboard
        if writer is not None:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar("Train/BatchLoss", loss.item(), global_step)
            writer.add_scalar("Train/LearningRate", optimizer.param_groups[0]["lr"], global_step)
        
        # 打印日志
        if batch_idx % TRAIN_CONFIG["log_interval"] == 0:
            print(
                f"Epoch {epoch}, Batch {batch_idx}, "
                f"Loss: {loss.item():.4f}"
            )
    
    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    """主训练函数"""
    print("=" * 50)
    print("开始训练扩散模型")
    print("=" * 50)
    
    # 设置设备
    device = torch.device(DEVICE)
    print(f"使用设备: {device}")
    
    # 创建扩散模型
    diffusion = GaussianDiffusion(
        timesteps=DIFFUSION_CONFIG["timesteps"],
        beta_schedule=DIFFUSION_CONFIG["beta_schedule"],
        beta_start=DIFFUSION_CONFIG["beta1"],
        beta_end=DIFFUSION_CONFIG["beta2"]
    )
    print(f"扩散模型配置: {DIFFUSION_CONFIG}")
    
    # 创建U-Net模型
    model = ContextUNet(
        in_channels=MODEL_CONFIG["in_channels"],
        n_feat=MODEL_CONFIG["n_feat"],
        n_cfeat=MODEL_CONFIG["n_cfeat"],
        height=MODEL_CONFIG["height"]
    ).to(device)
    
    # 计算模型参数数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数数量: {num_params:,}")
    
    # 创建数据加载器
    dataloader = get_dataloader(
        sprite_path=str(SPRITE_DATA_PATH),
        label_path=str(SPRITE_LABELS_PATH),
        batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=True,
        num_workers=TRAIN_CONFIG["num_workers"],
        normalize=True,
        null_context=False
    )
    print(f"数据集大小: {len(dataloader.dataset)}")
    print(f"批次大小: {TRAIN_CONFIG['batch_size']}")
    
    # 创建优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=TRAIN_CONFIG["learning_rate"]
    )
    print(f"学习率: {TRAIN_CONFIG['learning_rate']}")
    
    # 创建TensorBoard writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = TENSORBOARD_DIR / f"train_{timestamp}"
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard日志目录: {log_dir}")
    
    # 记录模型结构
    dummy_input = torch.randn(1, 3, 16, 16).to(device)
    dummy_t = torch.rand(1, 1).to(device)
    try:
        writer.add_graph(model, (dummy_input, dummy_t))
    except Exception as e:
        print(f"无法记录模型图: {e}")
    
    # 训练循环
    best_loss = float("inf")
    for epoch in range(TRAIN_CONFIG["n_epoch"]):
        # 线性衰减学习率
        lr = TRAIN_CONFIG["learning_rate"] * (1 - epoch / TRAIN_CONFIG["n_epoch"])
        optimizer.param_groups[0]["lr"] = lr
        
        # 训练一个epoch
        avg_loss = train_epoch(
            model=model,
            dataloader=dataloader,
            diffusion=diffusion,
            optimizer=optimizer,
            epoch=epoch,
            device=device,
            writer=writer,
            use_context=False  # 无条件训练
        )
        
        # 记录epoch级别的指标
        writer.add_scalar("Train/EpochLoss", avg_loss, epoch)
        writer.add_scalar("Train/LearningRate", lr, epoch)
        
        print(f"Epoch {epoch}/{TRAIN_CONFIG['n_epoch']-1}, "
              f"平均损失: {avg_loss:.4f}, 学习率: {lr:.6f}")
        
        # 保存模型
        if epoch % TRAIN_CONFIG["save_interval"] == 0 or epoch == TRAIN_CONFIG["n_epoch"] - 1:
            save_path = WEIGHTS_DIR / f"model_{epoch}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"模型已保存: {save_path}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = WEIGHTS_DIR / "model_best.pth"
                torch.save(model.state_dict(), best_path)
                print(f"最佳模型已保存: {best_path}")
    
    writer.close()
    print("=" * 50)
    print("训练完成！")
    print(f"TensorBoard日志: tensorboard --logdir {TENSORBOARD_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()

