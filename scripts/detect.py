#!/usr/bin/env python3
"""
图像篡改像素级分割测试/推理脚本

功能：
1. 加载训练好的模型
2. 对单张图片进行像素级篡改检测
3. 批量测试评估
4. 可视化结果输出
"""

import os
import sys
import pickle
import json
import numpy as np
import cv2
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from tqdm import tqdm


class ForgeryDetector:
    """图像篡改检测器"""
    
    def __init__(self, model_path: str):
        """
        初始化检测器
        
        Args:
            model_path: 模型文件路径
        """
        self.model_path = Path(model_path)
        self.load_model()
    
    def load_model(self):
        """加载模型"""
        model_file = self.model_path / 'model.pkl'
        
        if not model_file.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_file}")
        
        with open(model_file, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.scaler = data['scaler']
        self.threshold = data.get('threshold', 0.5)
        self.config = data.get('config', {
            'window_size': 32,
            'stride': 16,
            'feature_dim': 35
        })
        
        self.window_size = self.config['window_size']
        self.stride = self.config['stride']
        self.half = self.window_size // 2
        
        print(f"模型加载成功: {model_file}")
        print(f"  窗口大小: {self.window_size}")
        print(f"  滑动步长: {self.stride}")
        print(f"  阈值: {self.threshold}")
    
    def extract_features(self, patch: np.ndarray) -> np.ndarray:
        """从图像块提取特征"""
        features = []
        
        # 转灰度
        if len(patch.shape) == 3:
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        else:
            gray = patch
        gray = gray.astype(np.float32)
        
        # DCT特征
        dct = cv2.dct(gray)
        dct_low = dct[:8, :8]
        dct_high = dct[8:, 8:]
        features.append(np.mean(np.abs(dct_low)))
        features.append(np.std(dct_low))
        features.append(np.mean(np.abs(dct_high)))
        features.append(np.std(dct_high))
        features.append(np.percentile(np.abs(dct_low), 95))
        features.append(np.percentile(np.abs(dct_high), 95))
        features.append(np.max(np.abs(dct_low)))
        features.append(np.sum(np.abs(dct_high)) / (np.sum(np.abs(dct)) + 1e-8))
        
        # ELA特征
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, encoded = cv2.imencode('.jpg', patch, encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        ela = np.abs(patch.astype(np.float32) - decoded.astype(np.float32))
        features.append(np.mean(ela))
        features.append(np.std(ela))
        features.append(np.percentile(ela.flatten(), 95))
        features.append(np.max(ela))
        
        # Noise特征
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = gray - blurred
        features.append(np.mean(np.abs(noise)))
        features.append(np.std(noise))
        features.append(np.percentile(np.abs(noise), 95))
        features.append(np.max(np.abs(noise)))
        
        # Edge特征
        edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sobel_x**2 + sobel_y**2)
        features.append(np.mean(edges > 0))
        features.append(np.mean(mag))
        features.append(np.std(mag))
        features.append(np.max(mag))
        features.append(np.percentile(mag, 95))
        features.append(np.sum(mag > np.percentile(mag, 90)) / mag.size)
        
        # 纹理特征
        diff_h = np.abs(gray[:, 1:] - gray[:, :-1])
        diff_v = np.abs(gray[1:, :] - gray[:-1, :])
        features.append(np.mean(diff_h))
        features.append(np.std(diff_h))
        features.append(np.mean(diff_v))
        features.append(np.std(diff_v))
        features.append(np.percentile(diff_h, 95))
        features.append(np.percentile(diff_v, 95))
        features.append(np.percentile(np.abs(gray[1:, 1:] - gray[:-1, :-1]), 95))
        features.append(np.std(gray))
        
        # Color特征
        if len(patch.shape) == 3:
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            features.append(np.std(hsv[:,:,0]))
            features.append(np.std(hsv[:,:,1]))
            features.append(np.std(hsv[:,:,2]))
            features.append(np.std(patch[:,:,0]))
            features.append(np.std(patch[:,:,1]))
        else:
            features.extend([0] * 5)
        
        return np.array(features, dtype=np.float32)
    
    def detect(self, image: np.ndarray) -> Dict:
        """
        检测图像中的篡改区域
        
        Args:
            image: 输入图像 (BGR)
            
        Returns:
            {
                'mask': 篡改掩码,
                'heatmap': 置信度热力图,
                'confidence': 整体篡改置信度
            }
        """
        h, w = image.shape[:2]
        
        # 创建输出
        heatmap = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)
        
        # 滑动窗口预测
        features_list = []
        positions = []
        
        for y in range(self.half, h - self.half, self.stride):
            for x in range(self.half, w - self.half, self.stride):
                patch = image[y-self.half:y+self.half, x-self.half:x+self.half]
                if patch.shape[0] != self.window_size or patch.shape[1] != self.window_size:
                    continue
                
                feat = self.extract_features(patch)
                features_list.append(feat)
                positions.append((y, x))
        
        if not features_list:
            return {
                'mask': np.zeros((h, w), dtype=np.uint8),
                'heatmap': heatmap,
                'confidence': 0.0
            }
        
        # 批量预测
        X = np.array(features_list)
        X_scaled = self.scaler.transform(X)
        proba = self.model.predict_proba(X_scaled)[:, 1]
        
        # 生成热力图
        for (y, x), p in zip(positions, proba):
            y1 = max(0, y - self.stride // 2)
            y2 = min(h, y + self.stride // 2)
            x1 = max(0, x - self.stride // 2)
            x2 = min(w, x + self.stride // 2)
            
            heatmap[y1:y2, x1:x2] += p
            count_map[y1:y2, x1:x2] += 1
        
        # 归一化
        mask = count_map > 0
        heatmap[mask] = heatmap[mask] / count_map[mask]
        
        # 二值化
        binary_mask = (heatmap > self.threshold).astype(np.uint8) * 255
        
        # 后处理
        binary_mask = self.postprocess(binary_mask)
        
        # 整体置信度
        confidence = np.mean(proba)
        
        return {
            'mask': binary_mask,
            'heatmap': heatmap,
            'confidence': float(confidence)
        }
    
    def postprocess(self, mask: np.ndarray, min_area: int = 100) -> np.ndarray:
        """
        后处理：形态学操作 + 连通域过滤
        
        Args:
            mask: 二值掩码
            min_area: 最小连通域面积
            
        Returns:
            处理后的掩码
        """
        # 开运算去除噪点
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 闭运算填充空洞
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 连通域过滤
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        result = np.zeros_like(mask)
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                result[labels == i] = 255
        
        return result
    
    def detect_from_file(self, image_path: str) -> Dict:
        """从文件检测"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        return self.detect(image)
    
    def save_results(self, 
                     result: Dict, 
                     image: np.ndarray,
                     output_dir: str, 
                     name: str):
        """
        保存检测结果
        
        Args:
            result: 检测结果
            image: 原图
            output_dir: 输出目录
            name: 文件名前缀
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存掩码
        cv2.imwrite(str(output_dir / f"{name}_mask.png"), result['mask'])
        
        # 保存热力图
        heatmap_vis = (result['heatmap'] * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
        cv2.imwrite(str(output_dir / f"{name}_heatmap.png"), heatmap_colored)
        
        # 保存叠加图
        overlay = image.copy()
        mask_colored = np.zeros_like(image)
        mask_colored[result['mask'] > 0] = [0, 0, 255]  # 红色标记篡改区域
        overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
        cv2.imwrite(str(output_dir / f"{name}_overlay.png"), overlay)


def evaluate(detector: ForgeryDetector, 
             data_dir: str, 
             split: str = 'test') -> Dict:
    """
    评估模型性能
    
    Args:
        detector: 检测器
        data_dir: 数据目录
        split: 数据集划分
        
    Returns:
        评估结果
    """
    data_dir = Path(data_dir)
    images_dir = data_dir / split / 'images'
    masks_dir = data_dir / split / 'masks'
    
    if not images_dir.exists():
        print(f"错误: 目录不存在 {images_dir}")
        return None
    
    image_files = list(images_dir.glob('*.jpg'))
    print(f"评估 {split} 集: {len(image_files)} 张图片")
    
    metrics = []
    
    for img_file in tqdm(image_files):
        mask_name = img_file.stem + '.png'
        mask_file = masks_dir / mask_name
        
        if not mask_file.exists():
            continue
        
        # 读取GT
        gt_mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            continue
        
        # 检测
        result = detector.detect_from_file(str(img_file))
        pred_mask = result['mask']
        
        # 调整尺寸
        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]))
        
        # 计算指标
        pred = (pred_mask > 127).astype(np.uint8)
        gt = (gt_mask > 127).astype(np.uint8)
        
        tp = np.sum((pred == 1) & (gt == 1))
        fp = np.sum((pred == 1) & (gt == 0))
        fn = np.sum((pred == 0) & (gt == 1))
        tn = np.sum((pred == 0) & (gt == 0))
        
        iou = tp / (tp + fp + fn + 1e-6)
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        
        metrics.append({
            'image': img_file.name,
            'iou': float(iou),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        })
    
    # 计算平均
    if not metrics:
        return None
    
    avg_results = {
        'num_images': len(metrics),
        'avg_iou': float(np.mean([m['iou'] for m in metrics])),
        'avg_precision': float(np.mean([m['precision'] for m in metrics])),
        'avg_recall': float(np.mean([m['recall'] for m in metrics])),
        'avg_f1': float(np.mean([m['f1'] for m in metrics])),
        'per_image': metrics
    }
    
    return avg_results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='图像篡改检测')
    parser.add_argument('--model', type=str, default='./results/model',
                        help='模型目录')
    parser.add_argument('--image', type=str, default=None,
                        help='单张图片路径')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='数据目录 (批量测试)')
    parser.add_argument('--split', type=str, default='test',
                        help='数据集划分')
    parser.add_argument('--output', type=str, default='./results/output',
                        help='输出目录')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("图像篡改像素级分割检测")
    print("=" * 60)
    
    # 加载模型
    detector = ForgeryDetector(args.model)
    
    # 单张图片检测
    if args.image:
        print(f"\n检测图片: {args.image}")
        result = detector.detect_from_file(args.image)
        
        image = cv2.imread(args.image)
        name = Path(args.image).stem
        
        detector.save_results(result, image, args.output, name)
        
        print(f"\n检测结果:")
        print(f"  篡改置信度: {result['confidence']:.2%}")
        print(f"  篡改像素比例: {np.mean(result['mask'] > 0):.2%}")
        print(f"\n结果已保存到: {args.output}")
    
    # 批量测试
    elif args.data_dir:
        print(f"\n批量测试: {args.data_dir}")
        results = evaluate(detector, args.data_dir, args.split)
        
        if results:
            print("\n" + "=" * 60)
            print("评估结果")
            print("=" * 60)
            print(f"图片数量: {results['num_images']}")
            print(f"平均 IoU: {results['avg_iou']:.4f}")
            print(f"平均 Precision: {results['avg_precision']:.4f}")
            print(f"平均 Recall: {results['avg_recall']:.4f}")
            print(f"平均 F1: {results['avg_f1']:.4f}")
            
            # 保存结果
            output_file = Path(args.output) / 'evaluation_results.json'
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n结果已保存: {output_file}")
    
    else:
        print("\n请指定 --image 或 --data-dir 参数")


if __name__ == '__main__':
    main()