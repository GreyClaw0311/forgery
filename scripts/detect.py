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
                     name: str,
                     gt_mask: np.ndarray = None):
        """
        保存检测结果
        
        Args:
            result: 检测结果
            image: 原图
            output_dir: 输出目录
            name: 文件名前缀
            gt_mask: Ground Truth掩码 (可选)
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
        
        # 保存轮廓可视化
        contour_vis = self.visualize_contours(image, result['mask'])
        cv2.imwrite(str(output_dir / f"{name}_contour.png"), contour_vis)
        
        # 如果有GT，保存对比图
        if gt_mask is not None:
            comparison = self.create_comparison(image, gt_mask, result['mask'])
            cv2.imwrite(str(output_dir / f"{name}_comparison.png"), comparison)
    
    def visualize_contours(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        可视化篡改区域轮廓
        
        Args:
            image: 原图
            mask: 预测掩码
            
        Returns:
            轮廓可视化图像
        """
        vis = image.copy()
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 绘制轮廓
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
        
        # 在轮廓周围添加半透明填充
        overlay = vis.copy()
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), -1)
        vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
        
        return vis
    
    def create_comparison(self, 
                          image: np.ndarray, 
                          gt_mask: np.ndarray, 
                          pred_mask: np.ndarray) -> np.ndarray:
        """
        创建并排对比图
        
        Args:
            image: 原图
            gt_mask: Ground Truth掩码
            pred_mask: 预测掩码
            
        Returns:
            并排对比图像 (原图 | GT | 预测 | 差异)
        """
        h, w = image.shape[:2]
        
        # 确保尺寸一致
        if gt_mask.shape != (h, w):
            gt_mask = cv2.resize(gt_mask, (w, h))
        if pred_mask.shape != (h, w):
            pred_mask = cv2.resize(pred_mask, (w, h))
        
        # GT可视化 (绿色)
        gt_vis = image.copy()
        gt_colored = np.zeros_like(image)
        gt_colored[gt_mask > 127] = [0, 255, 0]
        gt_vis = cv2.addWeighted(gt_vis, 0.7, gt_colored, 0.3, 0)
        
        # 预测可视化 (红色)
        pred_vis = image.copy()
        pred_colored = np.zeros_like(image)
        pred_colored[pred_mask > 127] = [0, 0, 255]
        pred_vis = cv2.addWeighted(pred_vis, 0.7, pred_colored, 0.3, 0)
        
        # 差异可视化
        diff_vis = image.copy()
        gt_bin = (gt_mask > 127).astype(np.uint8)
        pred_bin = (pred_mask > 127).astype(np.uint8)
        
        # TP: 绿色, FP: 红色, FN: 蓝色
        diff_colored = np.zeros_like(image)
        tp = (gt_bin == 1) & (pred_bin == 1)
        fp = (gt_bin == 0) & (pred_bin == 1)
        fn = (gt_bin == 1) & (pred_bin == 0)
        
        diff_colored[tp] = [0, 255, 0]      # TP: 绿色
        diff_colored[fp] = [0, 0, 255]      # FP: 红色
        diff_colored[fn] = [255, 0, 0]      # FN: 蓝色
        
        diff_vis = cv2.addWeighted(diff_vis, 0.5, diff_colored, 0.5, 0)
        
        # 添加标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        def add_label(img, label, color=(255, 255, 255)):
            img_copy = img.copy()
            cv2.rectangle(img_copy, (5, 5), (200, 35), (0, 0, 0), -1)
            cv2.putText(img_copy, label, (10, 28), font, font_scale, color, thickness)
            return img_copy
        
        image_labeled = add_label(image, "Original")
        gt_labeled = add_label(gt_vis, "Ground Truth", (0, 255, 0))
        pred_labeled = add_label(pred_vis, "Prediction", (0, 0, 255))
        diff_labeled = add_label(diff_vis, "Diff(TP/FP/FN)", (255, 255, 255))
        
        # 拼接
        comparison = np.hstack([image_labeled, gt_labeled, pred_labeled, diff_labeled])
        
        return comparison
    
    def visualize_batch_results(self,
                                results: List[Dict],
                                output_dir: str,
                                num_samples: int = 10):
        """
        可视化批量检测结果
        
        Args:
            results: 检测结果列表
            output_dir: 输出目录
            num_samples: 显示样本数
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 选择最佳和最差样本
        sorted_results = sorted(results, key=lambda x: x['f1'], reverse=True)
        
        best_samples = sorted_results[:num_samples//2]
        worst_samples = sorted_results[-num_samples//2:]
        
        samples = best_samples + worst_samples
        
        # 创建网格可视化
        for i, r in enumerate(samples):
            if 'comparison' in r and r['comparison'] is not None:
                cv2.imwrite(str(output_dir / f"sample_{i:02d}_f1_{r['f1']:.3f}.png"), 
                           r['comparison'])
        
        # 生成汇总图
        self.create_summary_grid(samples, output_dir)
    
    def create_summary_grid(self, samples: List[Dict], output_dir: Path):
        """
        创建汇总网格图
        
        Args:
            samples: 样本列表
            output_dir: 输出目录
        """
        if not samples:
            return
        
        # 收集所有对比图
        comparisons = []
        for r in samples:
            if 'comparison' in r and r['comparison'] is not None:
                # 缩小尺寸以适应网格
                comp = r['comparison']
                h, w = comp.shape[:2]
                new_h = 300
                new_w = int(w * new_h / h)
                comp = cv2.resize(comp, (new_w, new_h))
                comparisons.append(comp)
        
        if not comparisons:
            return
        
        # 垂直拼接
        grid = np.vstack(comparisons)
        cv2.imwrite(str(output_dir / "summary_grid.png"), grid)
        
        print(f"汇总图已保存: {output_dir / 'summary_grid.png'}")


def evaluate(detector: ForgeryDetector, 
             data_dir: str, 
             split: str = 'test',
             output_dir: str = None,
             save_visualizations: bool = True,
             max_vis_samples: int = 20) -> Dict:
    """
    评估模型性能
    
    Args:
        detector: 检测器
        data_dir: 数据目录
        split: 数据集划分
        output_dir: 可视化输出目录
        save_visualizations: 是否保存可视化结果
        max_vis_samples: 最大可视化样本数
        
    Returns:
        评估结果
    """
    data_dir = Path(data_dir)
    images_dir = data_dir / split / 'images'
    masks_dir = data_dir / split / 'masks'
    
    if not images_dir.exists():
        print(f"错误: 目录不存在 {images_dir}")
        return None
    
    # 创建可视化输出目录
    if output_dir and save_visualizations:
        vis_dir = Path(output_dir) / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_visualizations = False
        vis_dir = None
    
    image_files = list(images_dir.glob('*.jpg'))
    print(f"评估 {split} 集: {len(image_files)} 张图片")
    
    metrics = []
    vis_count = 0
    
    for img_file in tqdm(image_files, desc="评估中"):
        mask_name = img_file.stem + '.png'
        mask_file = masks_dir / mask_name
        
        if not mask_file.exists():
            continue
        
        # 读取图像和GT
        image = cv2.imread(str(img_file))
        gt_mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        
        if image is None or gt_mask is None:
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
        
        metric = {
            'image': img_file.name,
            'iou': float(iou),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
        metrics.append(metric)
        
        # 保存可视化结果
        if save_visualizations and vis_count < max_vis_samples:
            # 创建对比图
            comparison = detector.create_comparison(image, gt_mask, pred_mask)
            
            # 保存
            sample_dir = vis_dir / f"{img_file.stem}_f1_{f1:.3f}"
            sample_dir.mkdir(exist_ok=True)
            
            cv2.imwrite(str(sample_dir / "comparison.png"), comparison)
            cv2.imwrite(str(sample_dir / "original.png"), image)
            cv2.imwrite(str(sample_dir / "gt_mask.png"), gt_mask)
            cv2.imwrite(str(sample_dir / "pred_mask.png"), pred_mask)
            
            # 保存热力图
            heatmap_vis = (result['heatmap'] * 255).astype(np.uint8)
            heatmap_colored = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
            cv2.imwrite(str(sample_dir / "heatmap.png"), heatmap_colored)
            
            vis_count += 1
    
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
    
    # 创建汇总网格图
    if save_visualizations and vis_count > 0:
        create_evaluation_summary(detector, metrics, data_dir, split, vis_dir)
    
    return avg_results


def create_evaluation_summary(detector: ForgeryDetector,
                               metrics: List[Dict],
                               data_dir: Path,
                               split: str,
                               output_dir: Path):
    """
    创建评估汇总可视化
    
    Args:
        detector: 检测器
        metrics: 评估指标列表
        data_dir: 数据目录
        split: 数据集划分
        output_dir: 输出目录
    """
    # 按F1排序
    sorted_metrics = sorted(metrics, key=lambda x: x['f1'], reverse=True)
    
    # 选择最佳5个和最差5个
    best_5 = sorted_metrics[:5]
    worst_5 = sorted_metrics[-5:]
    samples = best_5 + worst_5
    
    images_dir = data_dir / split / 'images'
    masks_dir = data_dir / split / 'masks'
    
    comparison_list = []
    
    for m in samples:
        img_name = m['image']
        img_file = images_dir / img_name
        mask_file = masks_dir / img_name.replace('.jpg', '.png')
        
        if not img_file.exists() or not mask_file.exists():
            continue
        
        image = cv2.imread(str(img_file))
        gt_mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        
        if image is None or gt_mask is None:
            continue
        
        result = detector.detect_from_file(str(img_file))
        pred_mask = result['mask']
        
        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]))
        
        comparison = detector.create_comparison(image, gt_mask, pred_mask)
        
        # 添加F1分数标签
        h, w = comparison.shape[:2]
        cv2.rectangle(comparison, (w-150, 5), (w-5, 35), (0, 0, 0), -1)
        cv2.putText(comparison, f"F1: {m['f1']:.3f}", (w-145, 28), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 缩小尺寸
        new_h = 200
        new_w = int(w * new_h / h)
        comparison = cv2.resize(comparison, (new_w, new_h))
        
        comparison_list.append(comparison)
    
    if comparison_list:
        # 垂直拼接
        grid = np.vstack(comparison_list)
        
        # 添加标题
        title = np.zeros((40, grid.shape[1], 3), dtype=np.uint8)
        cv2.putText(title, f"Evaluation Summary: Best 5 + Worst 5 (Total {len(metrics)} images)", 
                   (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        grid_with_title = np.vstack([title, grid])
        cv2.imwrite(str(output_dir / "summary_grid.png"), grid_with_title)
        
        print(f"\n汇总可视化已保存: {output_dir / 'summary_grid.png'}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='图像篡改检测')
    parser.add_argument('--model', type=str, default='./results/model',
                        help='模型目录')
    parser.add_argument('--image', type=str, default=None,
                        help='单张图片路径')
    parser.add_argument('--gt', type=str, default=None,
                        help='单张图片的Ground Truth掩码路径')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='数据目录 (批量测试)')
    parser.add_argument('--split', type=str, default='test',
                        help='数据集划分')
    parser.add_argument('--output', type=str, default='./results/output',
                        help='输出目录')
    parser.add_argument('--no-vis', action='store_true',
                        help='不保存可视化结果')
    parser.add_argument('--max-vis', type=int, default=20,
                        help='最大可视化样本数')
    
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
        
        # 加载GT (如果有)
        gt_mask = None
        if args.gt:
            gt_mask = cv2.imread(args.gt, cv2.IMREAD_GRAYSCALE)
            print(f"加载Ground Truth: {args.gt}")
        
        # 保存结果
        detector.save_results(result, image, args.output, name, gt_mask)
        
        print(f"\n检测结果:")
        print(f"  篡改置信度: {result['confidence']:.2%}")
        print(f"  篡改像素比例: {np.mean(result['mask'] > 0):.2%}")
        
        # 如果有GT，计算指标
        if gt_mask is not None:
            pred = (result['mask'] > 127).astype(np.uint8)
            gt = (gt_mask > 127).astype(np.uint8)
            
            if pred.shape != gt.shape:
                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
            
            tp = np.sum((pred == 1) & (gt == 1))
            fp = np.sum((pred == 1) & (gt == 0))
            fn = np.sum((pred == 0) & (gt == 1))
            
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)
            iou = tp / (tp + fp + fn + 1e-6)
            
            print(f"\n与Ground Truth对比:")
            print(f"  IoU: {iou:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1: {f1:.4f}")
        
        print(f"\n可视化结果已保存到: {args.output}")
        print(f"  - {name}_mask.png: 篡改掩码")
        print(f"  - {name}_heatmap.png: 置信度热力图")
        print(f"  - {name}_overlay.png: 叠加显示")
        print(f"  - {name}_contour.png: 轮廓可视化")
        if gt_mask is not None:
            print(f"  - {name}_comparison.png: 并排对比图")
    
    # 批量测试
    elif args.data_dir:
        print(f"\n批量测试: {args.data_dir}")
        
        results = evaluate(
            detector, 
            args.data_dir, 
            args.split,
            output_dir=args.output if not args.no_vis else None,
            save_visualizations=not args.no_vis,
            max_vis_samples=args.max_vis
        )
        
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
            
            if not args.no_vis:
                print(f"\n可视化结果保存在: {args.output}/visualizations/")
    
    else:
        print("\n请指定 --image 或 --data-dir 参数")
        print("\n使用示例:")
        print("  单图检测: python detect.py --model ./results/model --image test.jpg --output ./output")
        print("  单图+GT:  python detect.py --model ./results/model --image test.jpg --gt mask.png")
        print("  批量测试: python detect.py --model ./results/model --data-dir ./data --split test")


if __name__ == '__main__':
    main()