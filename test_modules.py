"""
快速测试脚本：验证所有模块可以正确导入
"""
import sys

def test_imports():
    """测试所有模块导入"""
    print("测试模块导入...")
    
    try:
        from config import (
            DEVICE, DIFFUSION_CONFIG, MODEL_CONFIG, TRAIN_CONFIG,
            SPRITE_DATA_PATH, SPRITE_LABELS_PATH
        )
        print("✓ config.py 导入成功")
    except Exception as e:
        print(f"✗ config.py 导入失败: {e}")
        return False
    
    try:
        from dataset import SpriteDataset, get_dataloader, get_transform
        print("✓ dataset.py 导入成功")
    except Exception as e:
        print(f"✗ dataset.py 导入失败: {e}")
        return False
    
    try:
        from models import ContextUNet
        print("✓ models/unet.py 导入成功")
    except Exception as e:
        print(f"✗ models/unet.py 导入失败: {e}")
        return False
    
    try:
        from diffusion import GaussianDiffusion, linear_beta_schedule, cosine_beta_schedule
        print("✓ diffusion.py 导入成功")
    except Exception as e:
        print(f"✗ diffusion.py 导入失败: {e}")
        return False
    
    try:
        from utils import (
            unorm, norm_all, norm_torch, plot_grid,
            plot_sample, generate_test_context, save_samples
        )
        print("✓ utils.py 导入成功")
    except Exception as e:
        print(f"✗ utils.py 导入失败: {e}")
        return False
    
    print("\n所有模块导入成功！")
    return True


def test_model_creation():
    """测试模型创建"""
    print("\n测试模型创建...")
    
    try:
        import torch
        from models import ContextUNet
        from config import MODEL_CONFIG, DEVICE
        
        device = torch.device(DEVICE)
        model = ContextUNet(
            in_channels=MODEL_CONFIG["in_channels"],
            n_feat=MODEL_CONFIG["n_feat"],
            n_cfeat=MODEL_CONFIG["n_cfeat"],
            height=MODEL_CONFIG["height"]
        ).to(device)
        
        # 测试前向传播
        batch_size = 4
        x = torch.randn(batch_size, 3, 16, 16).to(device)
        t = torch.rand(batch_size, 1).to(device)
        c = torch.rand(batch_size, 5).to(device)
        
        output = model(x, t, c)
        assert output.shape == (batch_size, 3, 16, 16), f"输出形状错误: {output.shape}"
        
        print(f"✓ 模型创建成功，输出形状: {output.shape}")
        return True
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_diffusion():
    """测试扩散模型"""
    print("\n测试扩散模型...")
    
    try:
        import torch
        from diffusion import GaussianDiffusion
        from config import DIFFUSION_CONFIG
        
        diffusion = GaussianDiffusion(
            timesteps=DIFFUSION_CONFIG["timesteps"],
            beta_schedule=DIFFUSION_CONFIG["beta_schedule"],
            beta_start=DIFFUSION_CONFIG["beta1"],
            beta_end=DIFFUSION_CONFIG["beta2"]
        )
        
        # 测试前向扩散
        batch_size = 4
        x_start = torch.randn(batch_size, 3, 16, 16)
        t = torch.randint(1, diffusion.timesteps + 1, (batch_size,))
        x_noised = diffusion.q_sample(x_start, t)
        
        assert x_noised.shape == x_start.shape, "加噪后形状不匹配"
        print(f"✓ 扩散模型测试成功")
        return True
    except Exception as e:
        print(f"✗ 扩散模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("开始测试项目代码")
    print("=" * 50)
    
    success = True
    success &= test_imports()
    success &= test_model_creation()
    success &= test_diffusion()
    
    print("\n" + "=" * 50)
    if success:
        print("所有测试通过！✓")
    else:
        print("部分测试失败 ✗")
    print("=" * 50)

