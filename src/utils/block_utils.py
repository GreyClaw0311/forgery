"""
分块处理工具模块

用于将图像分块处理，支持重叠分块
"""

import numpy as np
import cv2
from typing import List, Tuple, Generator


class BlockProcessor:
    """图像分块处理器"""
    
    def __init__(self, block_size: int = 32, overlap: int = 0):
        """
        初始化分块处理器
        
        Args:
            block_size: 块大小
            overlap: 重叠像素数
        """
        self.block_size = block_size
        self.overlap = overlap
    
    def split(self, image: np.ndarray) -> Generator[Tuple[np.ndarray, int, int], None, None]:
        """
        将图像分割成块
        
        Args:
            image: 输入图像
            
        Yields:
            (block, row, col): 块及其位置
        """
        h, w = image.shape[:2]
        step = self.block_size - self.overlap
        
        for i in range(0, h - self.block_size + 1, step):
            for j in range(0, w - self.block_size + 1, step):
                if len(image.shape) == 3:
                    block = image[i:i+self.block_size, j:j+self.block_size, :]
                else:
                    block = image[i:i+self.block_size, j:j+self.block_size]
                yield block, i, j
    
    def get_block_positions(self, image: np.ndarray) -> List[Tuple[int, int]]:
        """获取所有块的位置"""
        positions = []
        h, w = image.shape[:2]
        step = self.block_size - self.overlap
        
        for i in range(0, h - self.block_size + 1, step):
            for j in range(0, w - self.block_size + 1, step):
                positions.append((i, j))
        
        return positions
    
    def merge_blocks(self, blocks: List[np.ndarray], 
                     positions: List[Tuple[int, int]],
                     image_shape: Tuple[int, int]) -> np.ndarray:
        """
        合并块为完整图像
        
        对于重叠区域，取平均值
        """
        h, w = image_shape
        result = np.zeros((h, w), dtype=np.float64)
        count = np.zeros((h, w), dtype=np.float64)
        
        for block, (i, j) in zip(blocks, positions):
            bh, bw = block.shape[:2]
            result[i:i+bh, j:j+bw] += block
            count[i:i+bh, j:j+bw] += 1
        
        # 避免除零
        count[count == 0] = 1
        result /= count
        
        return result
    
    def get_neighbors(self, position: Tuple[int, int], 
                      positions: List[Tuple[int, int]],
                      radius: int = 1) -> List[Tuple[int, int]]:
        """
        获取邻近块的位置
        
        Args:
            position: 当前块位置
            positions: 所有块位置
            radius: 邻域半径
            
        Returns:
            邻近块位置列表
        """
        i, j = position
        step = self.block_size - self.overlap
        neighbors = []
        
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di * step, j + dj * step
                if (ni, nj) in positions:
                    neighbors.append((ni, nj))
        
        return neighbors


def sliding_window(image: np.ndarray, 
                   window_size: int, 
                   step: int) -> Generator[Tuple[np.ndarray, int, int], None, None]:
    """
    滑动窗口
    
    Args:
        image: 输入图像
        window_size: 窗口大小
        step: 步长
        
    Yields:
        (window, row, col)
    """
    h, w = image.shape[:2]
    
    for i in range(0, h - window_size + 1, step):
        for j in range(0, w - window_size + 1, step):
            if len(image.shape) == 3:
                window = image[i:i+window_size, j:j+window_size, :]
            else:
                window = image[i:i+window_size, j:j+window_size]
            yield window, i, j


def compute_block_features(image: np.ndarray, 
                           block_size: int = 32) -> np.ndarray:
    """
    计算每个块的基本特征
    
    Returns:
        features: (num_blocks, num_features) 特征矩阵
    """
    processor = BlockProcessor(block_size)
    features = []
    
    for block, i, j in processor.split(image):
        # 转灰度
        if len(block.shape) == 3:
            gray = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)
        else:
            gray = block
        
        # 计算特征
        feat = [
            np.mean(gray),           # 均值
            np.std(gray),            # 标准差
            np.var(gray),            # 方差
            np.min(gray),            # 最小值
            np.max(gray),            # 最大值
            np.median(gray),         # 中值
        ]
        features.append(feat)
    
    return np.array(features)


if __name__ == "__main__":
    # 测试
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    processor = BlockProcessor(block_size=32, overlap=0)
    
    blocks = list(processor.split(test_image))
    print(f"Total blocks: {len(blocks)}")
    print(f"Block shape: {blocks[0][0].shape}")
    
    positions = processor.get_block_positions(test_image)
    print(f"Positions: {len(positions)}")