"""
复制移动检测器

基于 Amerini et al. (2011) 的方法
使用SIFT特征匹配检测图像内部的重复区域
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from collections import defaultdict


class CopyMoveDetector:
    """
    复制移动检测器
    
    原理：
    - 复制移动篡改会在图像中产生相似区域
    - 通过特征点匹配检测重复区域
    - 使用SIFT/SURF等特征
    """
    
    def __init__(self, 
                 min_distance: int = 50,
                 ratio_thresh: float = 0.75,
                 min_matches: int = 4):
        """
        初始化复制移动检测器
        
        Args:
            min_distance: 最小匹配距离（排除自匹配）
            ratio_thresh: Lowe's ratio阈值
            min_matches: 最小匹配数
        """
        self.min_distance = min_distance
        self.ratio_thresh = ratio_thresh
        self.min_matches = min_matches
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """
        检测复制移动区域
        
        Args:
            image: 输入图像
            
        Returns:
            mask: 二值掩码 (0=正常, 255=篡改)
        """
        # 转灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # SIFT特征提取
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        
        if descriptors is None or len(keypoints) < 2:
            return np.zeros(gray.shape, dtype=np.uint8)
        
        # 特征匹配
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors, descriptors, k=2)
        
        # 过滤匹配
        good_pairs = []
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < self.ratio_thresh * n.distance:
                    kp1 = keypoints[m.queryIdx]
                    kp2 = keypoints[m.trainIdx]
                    
                    # 排除自匹配
                    dist = np.sqrt((kp1.pt[0] - kp2.pt[0])**2 + 
                                  (kp1.pt[1] - kp2.pt[1])**2)
                    if dist > self.min_distance:
                        good_pairs.append((kp1, kp2))
        
        if len(good_pairs) < self.min_matches:
            return np.zeros(gray.shape, dtype=np.uint8)
        
        # 生成掩码
        mask = np.zeros(gray.shape, dtype=np.uint8)
        
        for kp1, kp2 in good_pairs:
            # 在匹配点周围画圆
            cv2.circle(mask, (int(kp1.pt[0]), int(kp1.pt[1])), 15, 255, -1)
            cv2.circle(mask, (int(kp2.pt[0]), int(kp2.pt[1])), 15, 255, -1)
        
        # 后处理
        mask = self._post_process(mask)
        
        return mask
    
    def _post_process(self, mask: np.ndarray) -> np.ndarray:
        """
        后处理：形态学操作连接邻近区域
        
        Args:
            mask: 原始掩码
            
        Returns:
            processed_mask: 处理后的掩码
        """
        # 闭运算连接邻近区域
        kernel_close = np.ones((11, 11), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        
        # 开运算去除噪点
        kernel_open = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        
        return mask
    
    def detect_with_visualization(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        检测并返回可视化的匹配结果
        
        Args:
            image: 输入图像
            
        Returns:
            (mask, vis_image)
        """
        # 转灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # SIFT特征提取
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        
        vis_image = image.copy()
        
        if descriptors is None or len(keypoints) < 2:
            return np.zeros(gray.shape, dtype=np.uint8), vis_image
        
        # 特征匹配
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors, descriptors, k=2)
        
        good_pairs = []
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < self.ratio_thresh * n.distance:
                    kp1 = keypoints[m.queryIdx]
                    kp2 = keypoints[m.trainIdx]
                    
                    dist = np.sqrt((kp1.pt[0] - kp2.pt[0])**2 + 
                                  (kp1.pt[1] - kp2.pt[1])**2)
                    if dist > self.min_distance:
                        good_pairs.append((kp1, kp2))
        
        # 可视化匹配
        for kp1, kp2 in good_pairs:
            # 绘制连接线
            cv2.line(vis_image, 
                    (int(kp1.pt[0]), int(kp1.pt[1])),
                    (int(kp2.pt[0]), int(kp2.pt[1])),
                    (0, 255, 0), 1)
            # 绘制特征点
            cv2.circle(vis_image, (int(kp1.pt[0]), int(kp1.pt[1])), 5, (0, 0, 255), -1)
            cv2.circle(vis_image, (int(kp2.pt[0]), int(kp2.pt[1])), 5, (0, 0, 255), -1)
        
        # 生成掩码
        mask = np.zeros(gray.shape, dtype=np.uint8)
        
        if len(good_pairs) >= self.min_matches:
            for kp1, kp2 in good_pairs:
                cv2.circle(mask, (int(kp1.pt[0]), int(kp1.pt[1])), 15, 255, -1)
                cv2.circle(mask, (int(kp2.pt[0]), int(kp2.pt[1])), 15, 255, -1)
            
            mask = self._post_process(mask)
        
        return mask, vis_image
    
    def cluster_matches(self, good_pairs: List[Tuple], 
                        image_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        聚类匹配点对，检测多个复制区域
        
        Args:
            good_pairs: 匹配点对列表
            image_shape: 图像形状
            
        Returns:
            (mask1, mask2): 源区域掩码和目标区域掩码
        """
        if len(good_pairs) < self.min_matches:
            return (np.zeros(image_shape, dtype=np.uint8),
                    np.zeros(image_shape, dtype=np.uint8))
        
        # 提取点坐标
        points1 = np.array([(kp1.pt[0], kp1.pt[1]) for kp1, kp2 in good_pairs])
        points2 = np.array([(kp2.pt[0], kp2.pt[1]) for kp1, kp2 in good_pairs])
        
        # 使用简单的聚类（基于距离）
        # 这里简化处理，直接生成掩码
        
        mask1 = np.zeros(image_shape, dtype=np.uint8)
        mask2 = np.zeros(image_shape, dtype=np.uint8)
        
        for kp1, kp2 in good_pairs:
            cv2.circle(mask1, (int(kp1.pt[0]), int(kp1.pt[1])), 15, 255, -1)
            cv2.circle(mask2, (int(kp2.pt[0]), int(kp2.pt[1])), 15, 255, -1)
        
        mask1 = self._post_process(mask1)
        mask2 = self._post_process(mask2)
        
        return mask1, mask2
    
    def get_heatmap(self, image: np.ndarray) -> np.ndarray:
        """
        生成热力图（连续值）
        
        Args:
            image: 输入图像
            
        Returns:
            heatmap: 热力图 (0-1)
        """
        # 转灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # SIFT特征提取
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        
        if descriptors is None or len(keypoints) < 2:
            return np.zeros(gray.shape, dtype=float)
        
        # 特征匹配
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors, descriptors, k=2)
        
        # 统计每个特征点的匹配次数
        match_count = defaultdict(int)
        
        for match in matches:
            if len(match) == 2:
                m, n = match
                if m.distance < self.ratio_thresh * n.distance:
                    kp1 = keypoints[m.queryIdx]
                    kp2 = keypoints[m.trainIdx]
                    
                    dist = np.sqrt((kp1.pt[0] - kp2.pt[0])**2 + 
                                  (kp1.pt[1] - kp2.pt[1])**2)
                    if dist > self.min_distance:
                        match_count[m.queryIdx] += 1
                        match_count[m.trainIdx] += 1
        
        # 生成热力图
        heatmap = np.zeros(gray.shape, dtype=float)
        
        for idx, kp in enumerate(keypoints):
            if idx in match_count:
                intensity = min(match_count[idx] / 10.0, 1.0)
                cv2.circle(heatmap, (int(kp.pt[0]), int(kp.pt[1])), 
                          int(kp.size), intensity, -1)
        
        # 平滑
        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
        
        # 归一化
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        return heatmap


def detect_copy_move(image_path: str,
                     output_path: Optional[str] = None) -> np.ndarray:
    """
    便捷函数：从文件路径检测
    
    Args:
        image_path: 图像路径
        output_path: 输出路径
        
    Returns:
        mask: 二值掩码
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    detector = CopyMoveDetector()
    mask = detector.detect(image)
    
    if output_path:
        cv2.imwrite(output_path, mask)
    
    return mask


if __name__ == "__main__":
    # 测试：创建一个简单的复制移动图像
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # 复制一个区域
    source = test_image[50:100, 50:100].copy()
    test_image[150:200, 150:200] = source
    
    detector = CopyMoveDetector()
    mask, vis = detector.detect_with_visualization(test_image)
    
    print(f"检测到篡改区域比例: {np.sum(mask > 0) / mask.size * 100:.2f}%")