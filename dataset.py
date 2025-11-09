"""
数据集加载和预处理模块
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from typing import Optional, Tuple
from config import DATASET_CONFIG


class SpriteDataset(Dataset):
    """
    Sprite数据集类
    
    数据集包含5类像素图像：
    - hero: 英雄角色
    - non-hero: 非英雄角色
    - food: 食物
    - spell: 法术
    - side-facing: 侧向角色
    """
    
    def __init__(
        self,
        sprite_path: str,
        label_path: str,
        transform: Optional[transforms.Compose] = None,
        null_context: bool = False
    ):
        """
        初始化数据集
        
        Args:
            sprite_path: Sprite图像数据路径
            label_path: 标签数据路径
            transform: 数据变换
            null_context: 是否使用空条件
        """
        self.sprites = np.load(sprite_path)
        self.labels = np.load(label_path)
        self.transform = transform
        self.null_context = null_context
        
        print(f"Sprite数据形状: {self.sprites.shape}")
        print(f"标签数据形状: {self.labels.shape}")
        print(f"数据范围: [{self.sprites.min()}, {self.sprites.max()}]")
        
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.sprites)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            (image, label): 图像和标签元组
        """
        image = self.sprites[idx]
        
        if self.transform:
            image = self.transform(image)
        
        if self.null_context:
            label = torch.zeros(5, dtype=torch.float32)
        else:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return image, label
    
    def get_shapes(self) -> Tuple[Tuple, Tuple]:
        """返回数据和标签的形状"""
        return self.sprites.shape, self.labels.shape


def get_transform(normalize: bool = True) -> transforms.Compose:
    """
    获取数据变换
    
    Args:
        normalize: 是否归一化到[-1, 1]
        
    Returns:
        transform: 数据变换组合
    """
    transform_list = [
        transforms.ToTensor(),  # 从[0,255]转换到[0.0,1.0]，并转换为CHW格式
    ]
    
    if normalize:
        # 归一化到[-1, 1]
        transform_list.append(
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        )
    
    return transforms.Compose(transform_list)


def get_dataloader(
    sprite_path: str,
    label_path: str,
    batch_size: int = 100,
    shuffle: bool = True,
    num_workers: int = 4,
    normalize: bool = True,
    null_context: bool = False
) -> DataLoader:
    """
    获取数据加载器
    
    Args:
        sprite_path: Sprite图像数据路径
        label_path: 标签数据路径
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 数据加载进程数
        normalize: 是否归一化
        null_context: 是否使用空条件
        
    Returns:
        dataloader: 数据加载器
    """
    transform = get_transform(normalize=normalize)
    dataset = SpriteDataset(
        sprite_path=sprite_path,
        label_path=label_path,
        transform=transform,
        null_context=null_context
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader

