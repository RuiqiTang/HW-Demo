"""
扩散模型核心实现
支持DDPM和DDIM采样方法
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List
import math


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """
    线性beta调度
    
    Args:
        timesteps: 时间步数
        beta_start: beta起始值
        beta_end: beta结束值
        
    Returns:
        betas: beta序列
    """
    scale = 1000.0 / timesteps
    beta_start = scale * beta_start
    beta_end = scale * beta_end
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    余弦beta调度
    
    Args:
        timesteps: 时间步数
        s: 偏移参数
        
    Returns:
        betas: beta序列
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps + s) / (1 + s) * math.pi * 0.5) ** 2)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0, 0.999)


class GaussianDiffusion:
    """
    高斯扩散模型
    
    实现DDPM和DDIM采样方法
    """
    
    def __init__(
        self,
        timesteps: int = 500,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02
    ):
        """
        初始化扩散模型
        
        Args:
            timesteps: 扩散步数
            beta_schedule: beta调度方式 ('linear' 或 'cosine')
            beta_start: beta起始值
            beta_end: beta结束值
        """
        self.timesteps = timesteps
        
        # 设置beta调度
        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")
        
        self.betas = betas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 前向扩散计算
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # 反向扩散计算
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
    
    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: Tuple) -> torch.Tensor:
        """
        从张量a中提取对应时间步t的值
        
        Args:
            a: 系数张量
            t: 时间步索引
            x_shape: 目标形状
            
        Returns:
            out: 提取的值
        """
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out
    
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向扩散过程：q(x_t|x_0)
        
        Args:
            x_start: 原始图像
            t: 时间步
            noise: 噪声（可选）
            
        Returns:
            x_t: 加噪后的图像
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def q_mean_variance(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算q(x_t|x_0)的均值和方差
        
        Args:
            x_start: 原始图像
            t: 时间步
            
        Returns:
            mean: 均值
            var: 方差
            log_var: 对数方差
        """
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        var = self._extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_var = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, var, log_var
    
    def q_posterior_mean_variance(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算q(x_{t-1}|x_t, x_0)的均值和方差
        
        Args:
            x_start: 原始图像
            x_t: 时间步t的图像
            t: 时间步
            
        Returns:
            post_mean: 后验均值
            post_var: 后验方差
            post_log_var: 后验对数方差
        """
        post_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        post_var = self._extract(self.posterior_variance, t, x_t.shape)
        post_log_var_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return post_mean, post_var, post_log_var_clipped
    
    def predict_start_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor
    ) -> torch.Tensor:
        """
        从噪声预测原始图像
        
        Args:
            x_t: 时间步t的图像
            t: 时间步
            noise: 预测的噪声
            
        Returns:
            x_start: 预测的原始图像
        """
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    def p_mean_variance(
        self,
        model: torch.nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        clip_denoised: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算p(x_{t-1}|x_t)的均值和方差
        
        Args:
            model: 去噪模型
            x_t: 时间步t的图像
            t: 时间步（整数索引）
            c: 条件标签（可选）
            clip_denoised: 是否裁剪去噪结果
            
        Returns:
            model_mean: 模型预测的均值
            model_var: 模型预测的方差
            model_log_var: 模型预测的对数方差
        """
        # 将整数时间步转换为归一化的浮点数（模型期望的格式）
        t_norm = t.float() / self.timesteps
        t_norm = t_norm.unsqueeze(1)  # [B] -> [B, 1]
        
        # 预测噪声（使用归一化时间步）
        if c is not None:
            pred_noise = model(x_t, t_norm, c)
        else:
            pred_noise = model(x_t, t_norm)
        
        # 预测原始图像（使用整数时间步进行扩散计算）
        x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
        
        if clip_denoised:
            x_recon = torch.clamp(x_recon, min=-1.0, max=1.0)
        
        # 计算后验分布
        model_mean, post_var, post_log_var = self.q_posterior_mean_variance(x_recon, x_t, t)
        
        return model_mean, post_var, post_log_var
    
    @torch.no_grad()
    def p_sample(
        self,
        model: torch.nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        clip_denoised: bool = True
    ) -> torch.Tensor:
        """
        DDPM采样：从p(x_{t-1}|x_t)采样
        
        Args:
            model: 去噪模型
            x_t: 时间步t的图像
            t: 时间步
            c: 条件标签（可选）
            clip_denoised: 是否裁剪去噪结果
            
        Returns:
            pred_img: 预测的图像
        """
        model_mean, _, model_log_var = self.p_mean_variance(
            model, x_t, t, c, clip_denoised
        )
        
        noise = torch.randn_like(x_t)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
        
        pred_img = model_mean + nonzero_mask * (0.5 * model_log_var).exp() * noise
        
        return pred_img
    
    @torch.no_grad()
    def p_sample_loop(
        self,
        model: torch.nn.Module,
        shape: Tuple[int, ...],
        c: Optional[torch.Tensor] = None,
        clip_denoised: bool = True,
        return_all_timesteps: bool = False
    ) -> List[torch.Tensor]:
        """
        DDPM采样循环
        
        Args:
            model: 去噪模型
            shape: 图像形状 [B, C, H, W]
            c: 条件标签（可选）
            clip_denoised: 是否裁剪去噪结果
            return_all_timesteps: 是否返回所有时间步
            
        Returns:
            images: 生成的图像列表
        """
        batch_size = shape[0]
        device = next(model.parameters()).device
        
        img = torch.randn(shape, device=device)
        img_list = []
        
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, c, clip_denoised)
            
            if return_all_timesteps or i % 50 == 0:
                img_list.append(img.cpu())
        
        return img_list
    
    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        image_size: int,
        batch_size: int = 8,
        channels: int = 3,
        c: Optional[torch.Tensor] = None,
        clip_denoised: bool = True,
        return_all_timesteps: bool = False
    ) -> List[torch.Tensor]:
        """
        DDPM采样接口
        
        Args:
            model: 去噪模型
            image_size: 图像尺寸
            batch_size: 批次大小
            channels: 通道数
            c: 条件标签（可选）
            clip_denoised: 是否裁剪去噪结果
            return_all_timesteps: 是否返回所有时间步
            
        Returns:
            images: 生成的图像列表
        """
        shape = (batch_size, channels, image_size, image_size)
        return self.p_sample_loop(
            model, shape, c, clip_denoised, return_all_timesteps
        )
    
    @torch.no_grad()
    def ddim_sample(
        self,
        model: torch.nn.Module,
        image_size: int,
        batch_size: int = 8,
        channels: int = 3,
        ddim_timesteps: int = 50,
        ddim_eta: float = 0.0,
        ddim_discr_method: str = "uniform",
        c: Optional[torch.Tensor] = None,
        clip_denoised: bool = True
    ) -> torch.Tensor:
        """
        DDIM采样
        
        Args:
            model: 去噪模型
            image_size: 图像尺寸
            batch_size: 批次大小
            channels: 通道数
            ddim_timesteps: DDIM采样步数
            ddim_eta: DDIM eta参数
            ddim_discr_method: 时间步离散化方法 ('uniform' 或 'quad')
            c: 条件标签（可选）
            clip_denoised: 是否裁剪去噪结果
            
        Returns:
            sample_img: 生成的图像
        """
        # 设置DDIM时间步序列
        if ddim_discr_method == "uniform":
            c_step = self.timesteps // ddim_timesteps
            ddim_timestep_seq = np.asarray(list(range(0, self.timesteps, c_step)))
        elif ddim_discr_method == "quad":
            ddim_timestep_seq = (
                np.linspace(0, np.sqrt(self.timesteps * 0.8), ddim_timesteps) ** 2
            ).astype(int)
        else:
            raise ValueError(f"Unknown ddim_discr_method: {ddim_discr_method}")
        
        ddim_timestep_seq = ddim_timestep_seq + 1
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])
        
        device = next(model.parameters()).device
        sample_img = torch.randn(
            (batch_size, channels, image_size, image_size),
            device=device
        )
        
        for i in reversed(range(0, len(ddim_timestep_seq))):
            t_int = torch.full(
                (batch_size,),
                ddim_timestep_seq[i],
                device=device,
                dtype=torch.long
            )
            t_norm = torch.full(
                (batch_size,),
                ddim_timestep_seq[i] / self.timesteps,
                device=device,
                dtype=torch.float32
            )
            prev_t = torch.full(
                (batch_size,),
                ddim_timestep_prev_seq[i],
                device=device,
                dtype=torch.long
            )
            
            # 提取alpha累积乘积
            alpha_cumprod_t = self._extract(
                self.alphas_cumprod, t_int, sample_img.shape
            )
            alpha_cumprod_t_prev = self._extract(
                self.alphas_cumprod, prev_t, sample_img.shape
            )
            
            # 预测噪声（使用归一化时间步）
            if c is not None:
                pred_noise = model(sample_img, t_norm.unsqueeze(1), c)
            else:
                pred_noise = model(sample_img, t_norm.unsqueeze(1))
            
            # 预测原始图像
            pred_x0 = (
                sample_img - torch.sqrt(1.0 - alpha_cumprod_t) * pred_noise
            ) / torch.sqrt(alpha_cumprod_t)
            
            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, min=-1.0, max=1.0)
            
            # 计算方差
            sigmas_t = ddim_eta * torch.sqrt(
                (1.0 - alpha_cumprod_t_prev) / (1.0 - alpha_cumprod_t) *
                (1 - alpha_cumprod_t / alpha_cumprod_t_prev)
            )
            
            # 生成x_{t-1}
            dir_point_to_x_t = torch.sqrt(
                1.0 - alpha_cumprod_t_prev - sigmas_t ** 2
            ) * pred_noise
            x_prev = (
                torch.sqrt(alpha_cumprod_t_prev) * pred_x0 +
                dir_point_to_x_t +
                sigmas_t * torch.randn_like(sample_img)
            )
            
            sample_img = x_prev
        
        return sample_img
    
    def train_losses(
        self,
        model: torch.nn.Module,
        x_start: torch.Tensor,
        t: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算训练损失
        
        Args:
            model: 去噪模型
            x_start: 原始图像
            t: 时间步（归一化到[0,1]）
            c: 条件标签（可选）
            noise: 噪声（可选）
            
        Returns:
            loss: MSE损失
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # 将归一化的时间步转换为整数索引
        t_int = (t * self.timesteps).long().clamp(0, self.timesteps - 1)
        x_noised = self.q_sample(x_start, t_int, noise)
        
        if c is not None:
            predicted_noise = model(x_noised, t.unsqueeze(1), c)
        else:
            predicted_noise = model(x_noised, t.unsqueeze(1))
        
        loss = F.mse_loss(noise, predicted_noise)
        return loss

