"""
配置文件：定义所有超参数和路径
"""
import os
from pathlib import Path

# 项目根目录
ROOT_DIR = Path(__file__).parent

# 数据路径
DATA_DIR = ROOT_DIR / "Diffusion_Demo"
SPRITE_DATA_PATH = DATA_DIR / "sprites_1788_16x16.npy"
SPRITE_LABELS_PATH = DATA_DIR / "sprite_labels_nc_1788_16x16.npy"

# 输出路径
OUTPUT_DIR = ROOT_DIR / "outputs"
WEIGHTS_DIR = OUTPUT_DIR / "weights"
RESULTS_DIR = OUTPUT_DIR / "results"
LOGS_DIR = OUTPUT_DIR / "logs"
TENSORBOARD_DIR = OUTPUT_DIR / "tensorboard"

# 创建输出目录
OUTPUT_DIR.mkdir(exist_ok=True)
WEIGHTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
TENSORBOARD_DIR.mkdir(exist_ok=True)

# 设备配置
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

# 扩散模型超参数
DIFFUSION_CONFIG = {
    "timesteps": 500,  # 扩散步数
    "beta1": 1e-4,     # beta起始值
    "beta2": 0.02,     # beta结束值
    "beta_schedule": "linear",  # beta调度方式: linear 或 cosine
}

# 网络超参数
MODEL_CONFIG = {
    "in_channels": 3,      # 输入通道数（RGB）
    "n_feat": 64,          # 特征维度
    "n_cfeat": 5,          # 条件特征维度（5类：hero, non-hero, food, spell, side-facing）
    "height": 16,          # 图像高度（16x16）
}

# 训练超参数
TRAIN_CONFIG = {
    "batch_size": 100,
    "n_epoch": 32,
    "learning_rate": 1e-3,
    "num_workers": 4,
    "save_interval": 4,    # 每N个epoch保存一次模型
    "log_interval": 100,   # 每N个batch打印一次日志
}

# 采样超参数
SAMPLING_CONFIG = {
    "n_sample": 32,        # 采样数量
    "ddim_timesteps": 50,  # DDIM采样步数
    "ddim_eta": 0.0,       # DDIM eta参数
    "clip_denoised": True, # 是否裁剪去噪结果
}

# 数据集配置
DATASET_CONFIG = {
    "normalize": True,     # 是否归一化到[-1, 1]
    "null_context": False, # 是否使用空条件
}

