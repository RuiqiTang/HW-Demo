"""
采样和生成脚本
支持DDPM和DDIM采样方法
"""
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from config import (
    DEVICE, MODEL_CONFIG, SAMPLING_CONFIG, DIFFUSION_CONFIG,
    WEIGHTS_DIR, RESULTS_DIR
)
from models import ContextUNet
from diffusion import GaussianDiffusion
from utils import plot_grid, plot_sample, save_samples, generate_test_context


def sample_ddpm(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    n_sample: int,
    image_size: int,
    channels: int,
    device: torch.device,
    c: torch.Tensor = None,
    save_rate: int = 20,
    return_all: bool = False
) -> tuple:
    """
    DDPM采样
    
    Args:
        model: 训练好的模型
        diffusion: 扩散模型
        n_sample: 采样数量
        image_size: 图像尺寸
        channels: 通道数
        device: 设备
        c: 条件标签（可选）
        save_rate: 保存频率
        return_all: 是否返回所有时间步
        
    Returns:
        samples: 最终样本
        intermediate: 中间过程（可选）
    """
    model.eval()
    
    with torch.no_grad():
        # 从噪声开始
        samples = torch.randn(n_sample, channels, image_size, image_size).to(device)
        intermediate = []
        
        for i in tqdm(reversed(range(0, diffusion.timesteps)), desc="DDPM采样"):
            t_int = torch.full((n_sample,), i, device=device, dtype=torch.long)
            
            # 去噪（p_sample内部会处理时间步归一化）
            samples = diffusion.p_sample(model, samples, t_int, c, clip_denoised=True)
            
            # 保存中间结果
            if return_all and (i % save_rate == 0 or i < 8):
                intermediate.append(samples.cpu().numpy())
        
        if return_all:
            intermediate = np.stack(intermediate)
            return samples, intermediate
        else:
            return samples, None


def sample_ddim(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    n_sample: int,
    image_size: int,
    channels: int,
    device: torch.device,
    c: torch.Tensor = None,
    ddim_timesteps: int = 50,
    ddim_eta: float = 0.0
) -> torch.Tensor:
    """
    DDIM采样
    
    Args:
        model: 训练好的模型
        diffusion: 扩散模型
        n_sample: 采样数量
        image_size: 图像尺寸
        channels: 通道数
        device: 设备
        c: 条件标签（可选）
        ddim_timesteps: DDIM采样步数
        ddim_eta: DDIM eta参数
        
    Returns:
        samples: 生成的样本
    """
    model.eval()
    
    with torch.no_grad():
        samples = diffusion.ddim_sample(
            model=model,
            image_size=image_size,
            batch_size=n_sample,
            channels=channels,
            ddim_timesteps=ddim_timesteps,
            ddim_eta=ddim_eta,
            c=c,
            clip_denoised=True
        )
        
        return samples


def main():
    """主采样函数"""
    print("=" * 50)
    print("开始采样")
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
    
    # 创建模型
    model = ContextUNet(
        in_channels=MODEL_CONFIG["in_channels"],
        n_feat=MODEL_CONFIG["n_feat"],
        n_cfeat=MODEL_CONFIG["n_cfeat"],
        height=MODEL_CONFIG["height"]
    ).to(device)
    
    # 加载模型权重
    weight_path = WEIGHTS_DIR / "model_best.pth"
    if not weight_path.exists():
        # 尝试加载最后一个epoch的模型
        weight_path = WEIGHTS_DIR / "model_31.pth"
    
    if weight_path.exists():
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"已加载模型权重: {weight_path}")
    else:
        print(f"警告: 未找到模型权重文件 {weight_path}")
        print("请先运行 train.py 训练模型")
        return
    
    model.eval()
    
    # DDPM采样
    print("\n" + "=" * 50)
    print("DDPM采样")
    print("=" * 50)
    
    n_sample = SAMPLING_CONFIG["n_sample"]
    samples_ddpm, intermediate_ddpm = sample_ddpm(
        model=model,
        diffusion=diffusion,
        n_sample=n_sample,
        image_size=MODEL_CONFIG["height"],
        channels=MODEL_CONFIG["in_channels"],
        device=device,
        c=None,
        save_rate=20,
        return_all=True
    )
    
    # 保存DDPM结果
    save_path_ddpm = RESULTS_DIR / "samples_ddpm.png"
    save_samples(samples_ddpm, save_path_ddpm, nrow=8)
    
    # 保存DDPM动画
    if intermediate_ddpm is not None:
        plot_sample(
            intermediate_ddpm,
            n_sample=n_sample,
            nrows=4,
            save_dir=RESULTS_DIR,
            filename="ddpm_animation",
            save=True
        )
    
    # DDIM采样
    print("\n" + "=" * 50)
    print("DDIM采样")
    print("=" * 50)
    
    samples_ddim = sample_ddim(
        model=model,
        diffusion=diffusion,
        n_sample=n_sample,
        image_size=MODEL_CONFIG["height"],
        channels=MODEL_CONFIG["in_channels"],
        device=device,
        c=None,
        ddim_timesteps=SAMPLING_CONFIG["ddim_timesteps"],
        ddim_eta=SAMPLING_CONFIG["ddim_eta"]
    )
    
    # 保存DDIM结果
    save_path_ddim = RESULTS_DIR / "samples_ddim.png"
    save_samples(samples_ddim, save_path_ddim, nrow=8)
    
    print("=" * 50)
    print("采样完成！")
    print("=" * 50)


def sample_conditional():
    """条件控制采样"""
    print("=" * 50)
    print("开始条件控制采样")
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
    
    # 创建模型
    model = ContextUNet(
        in_channels=MODEL_CONFIG["in_channels"],
        n_feat=MODEL_CONFIG["n_feat"],
        n_cfeat=MODEL_CONFIG["n_cfeat"],
        height=MODEL_CONFIG["height"]
    ).to(device)
    
    # 加载条件模型权重
    weight_path = WEIGHTS_DIR / "context_model_best.pth"
    if not weight_path.exists():
        weight_path = WEIGHTS_DIR / "context_model_31.pth"
    
    if weight_path.exists():
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"已加载模型权重: {weight_path}")
    else:
        print(f"警告: 未找到条件模型权重文件 {weight_path}")
        print("请先运行 train_conditional.py 训练条件模型")
        return
    
    model.eval()
    
    # 生成测试条件
    test_context = generate_test_context(n_cfeat=MODEL_CONFIG["n_cfeat"])
    test_context = test_context.to(device)
    
    print(f"测试条件形状: {test_context.shape}")
    print("条件类别: hero, non-hero, food, spell, side-facing, null")
    
    # 条件控制采样
    print("\n" + "=" * 50)
    print("条件控制DDPM采样")
    print("=" * 50)
    
    n_sample = test_context.shape[0]
    samples_conditional, _ = sample_ddpm(
        model=model,
        diffusion=diffusion,
        n_sample=n_sample,
        image_size=MODEL_CONFIG["height"],
        channels=MODEL_CONFIG["in_channels"],
        device=device,
        c=test_context,
        save_rate=20,
        return_all=False
    )
    
    # 保存条件采样结果
    save_path_conditional = RESULTS_DIR / "samples_conditional.png"
    save_samples(samples_conditional, save_path_conditional, nrow=6)
    
    print("=" * 50)
    print("条件控制采样完成！")
    print("=" * 50)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "conditional":
        sample_conditional()
    else:
        main()

