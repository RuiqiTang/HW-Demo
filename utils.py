"""
工具函数
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from torchvision.utils import save_image, make_grid
from pathlib import Path
from typing import List, Optional


def unorm(x: np.ndarray) -> np.ndarray:
    """
    归一化到[0,1]范围
    
    Args:
        x: 输入数组 (h, w, 3)
        
    Returns:
        normalized: 归一化后的数组
    """
    xmax = x.max((0, 1))
    xmin = x.min((0, 1))
    return (x - xmin) / (xmax - xmin + 1e-8)


def norm_all(store: np.ndarray, n_t: int, n_s: int) -> np.ndarray:
    """
    对所有时间步的所有样本进行归一化
    
    Args:
        store: 存储的数组
        n_t: 时间步数
        n_s: 样本数
        
    Returns:
        nstore: 归一化后的数组
    """
    nstore = np.zeros_like(store)
    for t in range(n_t):
        for s in range(n_s):
            nstore[t, s] = unorm(store[t, s])
    return nstore


def norm_torch(x_all: torch.Tensor) -> torch.Tensor:
    """
    对torch张量进行归一化
    
    Args:
        x_all: 输入张量 (n_samples, 3, h, w)
        
    Returns:
        normalized: 归一化后的张量
    """
    x = x_all.cpu().numpy()
    xmax = x.max((2, 3))
    xmin = x.min((2, 3))
    xmax = np.expand_dims(xmax, (2, 3))
    xmin = np.expand_dims(xmin, (2, 3))
    nstore = (x - xmin) / (xmax - xmin + 1e-8)
    return torch.from_numpy(nstore)


def plot_grid(
    x: torch.Tensor,
    n_sample: int,
    n_rows: int,
    save_dir: Path,
    filename: str,
    w: Optional[int] = None
) -> torch.Tensor:
    """
    绘制图像网格
    
    Args:
        x: 图像张量 (n_sample, 3, h, w)
        n_sample: 样本数量
        n_rows: 行数
        save_dir: 保存目录
        filename: 文件名
        w: 权重标识（可选）
        
    Returns:
        grid: 网格图像
    """
    ncols = n_sample // n_rows
    grid = make_grid(norm_torch(x), nrow=ncols)
    
    if w is not None:
        save_path = save_dir / f"{filename}_w{w}.png"
    else:
        save_path = save_dir / f"{filename}.png"
    
    save_image(grid, save_path)
    print(f"图像已保存: {save_path}")
    return grid


def plot_sample(
    x_gen_store: np.ndarray,
    n_sample: int,
    nrows: int,
    save_dir: Path,
    filename: str,
    w: Optional[int] = None,
    save: bool = True
) -> FuncAnimation:
    """
    绘制采样过程动画
    
    Args:
        x_gen_store: 生成的图像序列
        n_sample: 样本数量
        nrows: 行数
        save_dir: 保存目录
        filename: 文件名
        w: 权重标识（可选）
        save: 是否保存GIF
        
    Returns:
        ani: 动画对象
    """
    ncols = n_sample // nrows
    sx_gen_store = np.moveaxis(x_gen_store, 2, 4)  # 转换为numpy图像格式 (h,w,channels)
    nsx_gen_store = norm_all(
        sx_gen_store, sx_gen_store.shape[0], n_sample
    )  # 归一化到[0,1]
    
    # 创建动画
    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols,
        sharex=True, sharey=True,
        figsize=(ncols, nrows)
    )
    
    def animate_diff(i: int, store: np.ndarray):
        """动画帧函数"""
        print(f"GIF动画帧 {i}/{store.shape[0]}", end="\r")
        plots = []
        for row in range(nrows):
            for col in range(ncols):
                axs[row, col].clear()
                axs[row, col].set_xticks([])
                axs[row, col].set_yticks([])
                plots.append(
                    axs[row, col].imshow(store[i, (row * ncols) + col])
                )
        return plots
    
    ani = FuncAnimation(
        fig, animate_diff, fargs=[nsx_gen_store],
        interval=200, blit=False, repeat=True,
        frames=nsx_gen_store.shape[0]
    )
    plt.close()
    
    if save:
        if w is not None:
            save_path = save_dir / f"{filename}_w{w}.gif"
        else:
            save_path = save_dir / f"{filename}.gif"
        ani.save(save_path, dpi=100, writer=PillowWriter(fps=5))
        print(f"GIF已保存: {save_path}")
    
    return ani


def generate_test_context(n_cfeat: int = 5) -> torch.Tensor:
    """
    生成测试条件向量
    
    5类标签：
    - hero: [1,0,0,0,0]
    - non-hero: [0,1,0,0,0]
    - food: [0,0,1,0,0]
    - spell: [0,0,0,1,0]
    - side-facing: [0,0,0,0,1]
    - null: [0,0,0,0,0]
    
    Args:
        n_cfeat: 条件特征维度
        
    Returns:
        context: 条件向量
    """
    vec = torch.tensor([
        [1, 0, 0, 0, 0],  # hero
        [0, 1, 0, 0, 0],  # non-hero
        [0, 0, 1, 0, 0],  # food
        [0, 0, 0, 1, 0],  # spell
        [0, 0, 0, 0, 1],  # side-facing
        [0, 0, 0, 0, 0],  # null
    ] * 6)  # 重复6次，共36个样本
    
    return vec


def save_samples(
    samples: torch.Tensor,
    save_path: Path,
    nrow: int = 8
) -> None:
    """
    保存样本图像
    
    Args:
        samples: 样本张量 (n, 3, h, w)
        save_path: 保存路径
        nrow: 每行显示的图像数
    """
    # 归一化到[0,1]
    samples_norm = (samples + 1.0) / 2.0
    samples_norm = torch.clamp(samples_norm, 0.0, 1.0)
    
    grid = make_grid(samples_norm, nrow=nrow)
    save_image(grid, save_path)
    print(f"样本已保存: {save_path}")

