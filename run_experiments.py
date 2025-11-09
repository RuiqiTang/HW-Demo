"""
自动化实验脚本
运行所有实验要求的内容，包括训练、采样和结果生成
"""
import os
import sys
import time
import subprocess
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import json

from config import (
    DEVICE, DIFFUSION_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, SAMPLING_CONFIG,
    SPRITE_DATA_PATH, SPRITE_LABELS_PATH, WEIGHTS_DIR, RESULTS_DIR, TENSORBOARD_DIR
)
from dataset import SpriteDataset, get_transform
from models import ContextUNet
from diffusion import GaussianDiffusion
from utils import save_samples, generate_test_context
import sample


class ExperimentRunner:
    """实验运行器"""
    
    def __init__(self):
        self.device = torch.device(DEVICE)
        self.start_time = time.time()
        self.results = {
            "start_time": datetime.now().isoformat(),
            "experiments": []
        }
        
    def log(self, message: str, level: str = "INFO"):
        """打印日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")
    
    def run_step(self, step_name: str, func, *args, **kwargs):
        """运行一个实验步骤"""
        self.log(f"开始执行: {step_name}")
        step_start = time.time()
        
        try:
            result = func(*args, **kwargs)
            step_time = time.time() - step_start
            self.log(f"完成: {step_name} (耗时: {step_time:.2f}秒)")
            
            self.results["experiments"].append({
                "step": step_name,
                "status": "success",
                "time": step_time
            })
            
            return result
        except Exception as e:
            step_time = time.time() - step_start
            self.log(f"失败: {step_name} - {str(e)}", level="ERROR")
            
            self.results["experiments"].append({
                "step": step_name,
                "status": "failed",
                "error": str(e),
                "time": step_time
            })
            
            raise
    
    def check_data(self):
        """检查数据文件"""
        self.log("检查数据文件...")
        
        if not SPRITE_DATA_PATH.exists():
            raise FileNotFoundError(f"数据文件不存在: {SPRITE_DATA_PATH}")
        if not SPRITE_LABELS_PATH.exists():
            raise FileNotFoundError(f"标签文件不存在: {SPRITE_LABELS_PATH}")
        
        # 加载数据检查
        dataset = SpriteDataset(
            sprite_path=str(SPRITE_DATA_PATH),
            label_path=str(SPRITE_LABELS_PATH),
            transform=get_transform(normalize=True),
            null_context=False
        )
        
        self.log(f"数据集大小: {len(dataset)}")
        self.log(f"数据形状: {dataset.sprites.shape}")
        self.log(f"标签形状: {dataset.labels.shape}")
        
        return True
    
    def train_unconditional(self):
        """训练无条件模型"""
        self.log("开始训练无条件模型...")
        
        # 导入训练模块
        import train
        
        # 运行训练
        train.main()
        
        # 检查模型文件
        best_model = WEIGHTS_DIR / "model_best.pth"
        if not best_model.exists():
            # 尝试加载最后一个epoch的模型
            last_epoch = TRAIN_CONFIG["n_epoch"] - 1
            best_model = WEIGHTS_DIR / f"model_{last_epoch}.pth"
        
        if not best_model.exists():
            raise FileNotFoundError(f"训练后的模型文件不存在: {best_model}")
        
        self.log(f"模型已保存: {best_model}")
        return best_model
    
    def train_conditional(self):
        """训练条件模型"""
        self.log("开始训练条件控制模型...")
        
        # 导入训练模块
        import train_conditional
        
        # 运行训练
        train_conditional.main()
        
        # 检查模型文件
        best_model = WEIGHTS_DIR / "context_model_best.pth"
        if not best_model.exists():
            # 尝试加载最后一个epoch的模型
            last_epoch = TRAIN_CONFIG["n_epoch"] - 1
            best_model = WEIGHTS_DIR / f"context_model_{last_epoch}.pth"
        
        if not best_model.exists():
            raise FileNotFoundError(f"训练后的条件模型文件不存在: {best_model}")
        
        self.log(f"条件模型已保存: {best_model}")
        return best_model
    
    def sample_unconditional_ddpm(self, model_path: Path):
        """无条件DDPM采样"""
        self.log("开始DDPM采样（无条件）...")
        
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
        ).to(self.device)
        
        # 加载权重
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        
        # 采样
        n_sample = SAMPLING_CONFIG["n_sample"]
        samples, intermediate = sample.sample_ddpm(
            model=model,
            diffusion=diffusion,
            n_sample=n_sample,
            image_size=MODEL_CONFIG["height"],
            channels=MODEL_CONFIG["in_channels"],
            device=self.device,
            c=None,
            save_rate=20,
            return_all=True
        )
        
        # 保存结果
        save_path = RESULTS_DIR / "samples_ddpm.png"
        save_samples(samples, save_path, nrow=8)
        self.log(f"DDPM采样结果已保存: {save_path}")
        
        return samples, intermediate
    
    def sample_unconditional_ddim(self, model_path: Path):
        """无条件DDIM采样"""
        self.log("开始DDIM采样（无条件）...")
        
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
        ).to(self.device)
        
        # 加载权重
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        
        # 采样
        n_sample = SAMPLING_CONFIG["n_sample"]
        samples = sample.sample_ddim(
            model=model,
            diffusion=diffusion,
            n_sample=n_sample,
            image_size=MODEL_CONFIG["height"],
            channels=MODEL_CONFIG["in_channels"],
            device=self.device,
            c=None,
            ddim_timesteps=SAMPLING_CONFIG["ddim_timesteps"],
            ddim_eta=SAMPLING_CONFIG["ddim_eta"]
        )
        
        # 保存结果
        save_path = RESULTS_DIR / "samples_ddim.png"
        save_samples(samples, save_path, nrow=8)
        self.log(f"DDIM采样结果已保存: {save_path}")
        
        return samples
    
    def sample_conditional_all(self, model_path: Path):
        """条件控制采样（所有类别）"""
        self.log("开始条件控制采样...")
        
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
        ).to(self.device)
        
        # 加载权重
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        
        # 生成测试条件
        test_context = generate_test_context(n_cfeat=MODEL_CONFIG["n_cfeat"])
        test_context = test_context.to(self.device)
        
        # 为每个类别单独采样
        class_names = ["hero", "non-hero", "food", "spell", "side-facing", "null"]
        all_samples = []
        
        for i, class_name in enumerate(class_names):
            self.log(f"采样类别: {class_name}")
            class_context = test_context[i:i+1].repeat(6, 1)  # 每个类别采样6个
            
            samples, _ = sample.sample_ddpm(
                model=model,
                diffusion=diffusion,
                n_sample=6,
                image_size=MODEL_CONFIG["height"],
                channels=MODEL_CONFIG["in_channels"],
                device=self.device,
                c=class_context,
                save_rate=20,
                return_all=False
            )
            
            # 保存单个类别结果
            save_path = RESULTS_DIR / f"samples_conditional_{class_name}.png"
            save_samples(samples, save_path, nrow=6)
            self.log(f"类别 {class_name} 结果已保存: {save_path}")
            
            all_samples.append(samples)
        
        # 保存所有类别结果
        all_samples_tensor = torch.cat(all_samples, dim=0)
        save_path = RESULTS_DIR / "samples_conditional_all.png"
        save_samples(all_samples_tensor, save_path, nrow=6)
        self.log(f"所有条件采样结果已保存: {save_path}")
        
        return all_samples
    
    def generate_summary(self):
        """生成实验总结"""
        total_time = time.time() - self.start_time
        self.results["end_time"] = datetime.now().isoformat()
        self.results["total_time"] = total_time
        
        # 保存结果JSON
        summary_path = RESULTS_DIR / "experiment_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        # 打印总结
        self.log("=" * 60)
        self.log("实验总结")
        self.log("=" * 60)
        self.log(f"总耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
        self.log(f"成功步骤: {sum(1 for exp in self.results['experiments'] if exp['status'] == 'success')}")
        self.log(f"失败步骤: {sum(1 for exp in self.results['experiments'] if exp['status'] == 'failed')}")
        self.log(f"结果已保存: {summary_path}")
        self.log("=" * 60)
    
    def run_all(self, skip_training: bool = False):
        """运行所有实验"""
        self.log("=" * 60)
        self.log("开始自动化实验流程")
        self.log("=" * 60)
        
        try:
            # 1. 检查数据
            self.run_step("数据检查", self.check_data)
            
            # 2. 训练无条件模型
            if not skip_training:
                model_path = self.run_step("训练无条件模型", self.train_unconditional)
            else:
                model_path = WEIGHTS_DIR / "model_best.pth"
                if not model_path.exists():
                    model_path = WEIGHTS_DIR / f"model_{TRAIN_CONFIG['n_epoch']-1}.pth"
                self.log(f"跳过训练，使用已有模型: {model_path}")
            
            # 3. 训练条件模型
            if not skip_training:
                cond_model_path = self.run_step("训练条件模型", self.train_conditional)
            else:
                cond_model_path = WEIGHTS_DIR / "context_model_best.pth"
                if not cond_model_path.exists():
                    cond_model_path = WEIGHTS_DIR / f"context_model_{TRAIN_CONFIG['n_epoch']-1}.pth"
                self.log(f"跳过训练，使用已有条件模型: {cond_model_path}")
            
            # 4. DDPM采样
            if model_path.exists():
                self.run_step("DDPM采样", self.sample_unconditional_ddpm, model_path)
            
            # 5. DDIM采样
            if model_path.exists():
                self.run_step("DDIM采样", self.sample_unconditional_ddim, model_path)
            
            # 6. 条件控制采样
            if cond_model_path.exists():
                self.run_step("条件控制采样", self.sample_conditional_all, cond_model_path)
            
            # 7. 生成总结
            self.generate_summary()
            
            self.log("=" * 60)
            self.log("所有实验完成！")
            self.log("=" * 60)
            
        except Exception as e:
            self.log(f"实验过程中出现错误: {str(e)}", level="ERROR")
            import traceback
            traceback.print_exc()
            self.generate_summary()
            raise


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="自动化实验脚本")
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="跳过训练步骤，仅运行采样（需要已有模型）"
    )
    
    args = parser.parse_args()
    
    runner = ExperimentRunner()
    runner.run_all(skip_training=args.skip_training)


if __name__ == "__main__":
    main()

