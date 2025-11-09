"""
U-Net网络结构实现
支持时间步嵌入和条件控制
"""
import torch
import torch.nn as nn
from typing import Optional


class ResidualConvBlock(nn.Module):
    """残差卷积块"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        is_res: bool = False
    ) -> None:
        """
        初始化残差卷积块
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            is_res: 是否使用残差连接
        """
        super().__init__()
        
        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        
        # 第一个卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        
        # 第二个卷积层
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [B, C, H, W]
            
        Returns:
            out: 输出张量
        """
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            
            if self.same_channels:
                out = x + x2
            else:
                shortcut = nn.Conv2d(
                    x.shape[1], x2.shape[1],
                    kernel_size=1, stride=1, padding=0
                ).to(x.device)
                out = shortcut(x) + x2
            
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    """U-Net下采样块"""
    
    def __init__(self, in_channels: int, out_channels: int):
        """
        初始化下采样块
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
        """
        super(UnetDown, self).__init__()
        
        layers = [
            ResidualConvBlock(in_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
            nn.MaxPool2d(2)
        ]
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.model(x)


class UnetUp(nn.Module):
    """U-Net上采样块"""
    
    def __init__(self, in_channels: int, out_channels: int):
        """
        初始化上采样块
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
        """
        super(UnetUp, self).__init__()
        
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
            skip: 跳跃连接张量
            
        Returns:
            out: 输出张量
        """
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    """全连接嵌入层"""
    
    def __init__(self, input_dim: int, emb_dim: int):
        """
        初始化嵌入层
        
        Args:
            input_dim: 输入维度
            emb_dim: 嵌入维度
        """
        super(EmbedFC, self).__init__()
        
        self.input_dim = input_dim
        
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            out: 嵌入向量
        """
        x = x.view(-1, self.input_dim)
        return self.model(x)


class ContextUNet(nn.Module):
    """
    带条件控制的U-Net网络
    
    用于扩散模型的去噪网络，支持：
    - 时间步嵌入
    - 条件标签嵌入
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        n_feat: int = 64,
        n_cfeat: int = 5,
        height: int = 16
    ):
        """
        初始化ContextUNet
        
        Args:
            in_channels: 输入通道数
            n_feat: 特征维度
            n_cfeat: 条件特征维度
            height: 图像高度（假设为正方形）
        """
        super(ContextUNet, self).__init__()
        
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.h = height
        
        # 初始卷积层
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        
        # 下采样路径
        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        
        # 全局池化
        self.to_vec = nn.Sequential(nn.AvgPool2d((4)), nn.GELU())
        
        # 时间步嵌入
        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, 1 * n_feat)
        
        # 条件嵌入
        self.contextembed1 = EmbedFC(n_cfeat, 2 * n_feat)
        self.contextembed2 = EmbedFC(n_cfeat, 1 * n_feat)
        
        # 上采样路径
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, self.h // 4, self.h // 4),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )
        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        
        # 输出层
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像 [B, C, H, W]
            t: 时间步 [B, 1] 或 [B]
            c: 条件标签 [B, n_cfeat]，可选
            
        Returns:
            out: 预测的噪声 [B, C, H, W]
        """
        # 确保所有输入使用相同的数据类型
        dtype = x.dtype
        device = x.device
        
        # 处理时间步维度
        if t.dim() == 1:
            t = t.unsqueeze(1)
        t = t.to(dtype=dtype, device=device)
        
        # 处理条件输入
        if c is not None:
            c = c.to(dtype=dtype, device=device)
        
        # 初始卷积
        x = self.init_conv(x)
        
        # 下采样
        down1 = self.down1(x)
        down2 = self.down2(down1)
        
        # 全局池化
        hiddenvec = self.to_vec(down2)
        
        # 处理条件
        if c is None:
            c = torch.zeros(x.shape[0], self.n_cfeat, dtype=dtype, device=device)
        
        # 嵌入时间和条件
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)
        
        # 上采样
        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1 * up1 + temb1, down2)
        up3 = self.up2(cemb2 * up2 + temb2, down1)
        
        # 输出
        out = self.out(torch.cat((up3, x), 1))
        
        return out

