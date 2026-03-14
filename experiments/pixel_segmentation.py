#!/usr/bin/env python3
"""
像素级图像篡改分割系统

功能：
1. 滑动窗口提取像素级特征
2. 训练分类器预测篡改概率
3. 输出精确篡改区域Mask
"""

import os
import sys
from pathlib import Path
import numpy as np
import cv2
import pickle
import json
from datetime import datetime
from collections import defaultdict

# 简单进度条替代
def progress(iterable, desc=""):
    total = len(iterable) if hasattr(iterable, '__len__') else None
    for i, item in enumerate(iterable):
        if total and i % max(1, total // 10) == 0:
            print(f"\r{desc}: {i}/{total}", end='', flush=True)
        yield item
    if total:
        print(f"\r{desc}: {total}/{total} ✓")

# ML
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_image, load_mask, compute_iou, compute_precision_recall, compute_f1


# ============================================================================
# 特征提取器
# ============================================================================

class PixelFeatureExtractor:
    """像素级特征提取器"""
    
    def __init__(self, window_size=32, stride=8):
        self.window_size = window_size
        self.stride = stride
    
    def extract_dct_features(self, patch):
        """DCT特征"""
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if len(patch.shape) == 3 else patch
        gray = gray.astype(np.float32)
        
        # DCT变换
        dct = cv2.dct(gray)
        
        # 取低频系数
        features = []
        features.append(np.mean(np.abs(dct[:8, :8])))
        features.append(np.std(dct[:8, :8]))
        features.append(np.mean(np.abs(dct[8:, 8:])))  # 高频
        features.append(np.std(dct[8:, 8:]))
        
        # DCT能量分布
        total_energy = np.sum(dct ** 2)
        low_energy = np.sum(dct[:4, :4] ** 2)
        features.append(low_energy / (total_energy + 1e-8))
        
        return features
    
    def extract_ela_features(self, patch, quality=90):
        """ELA特征"""
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if len(patch.shape) == 3 else patch
        
        # 编码解码
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', cv2.cvtColor(patch, cv2.COLOR_BGR2RGB) if len(patch.shape) == 3 else gray, encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if len(patch.shape) == 3 else cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        
        # 计算差异
        diff = cv2.absdiff(patch, decoded)
        if len(diff.shape) == 3:
            diff = np.mean(diff, axis=2)
        
        features = [
            np.mean(diff),
            np.std(diff),
            np.max(diff),
            np.percentile(diff, 95),
        ]
        return features
    
    def extract_noise_features(self, patch):
        """噪声特征"""
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if len(patch.shape) == 3 else patch
        gray = gray.astype(np.float32)
        
        # 高斯滤波去噪
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = gray - blurred
        
        features = [
            np.mean(np.abs(noise)),
            np.std(noise),
            np.var(noise),
            np.percentile(np.abs(noise), 95),
        ]
        return features
    
    def extract_edge_features(self, patch):
        """边缘特征"""
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if len(patch.shape) == 3 else patch
        
        # Canny边缘
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.mean(edges > 0)
        
        # Sobel梯度
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        features = [
            edge_density,
            np.mean(magnitude),
            np.std(magnitude),
            np.max(magnitude),
        ]
        return features
    
    def extract_lbp_features(self, patch, num_points=24, radius=3):
        """LBP纹理特征"""
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if len(patch.shape) == 3 else patch
        
        # 简化版LBP
        lbp = np.zeros_like(gray, dtype=np.float32)
        for i in range(radius, gray.shape[0] - radius):
            for j in range(radius, gray.shape[1] - radius):
                center = gray[i, j]
                code = 0
                for k in range(num_points):
                    angle = 2 * np.pi * k / num_points
                    x = int(i + radius * np.cos(angle))
                    y = int(j + radius * np.sin(angle))
                    if 0 <= x < gray.shape[0] and 0 <= y < gray.shape[1]:
                        if gray[x, y] >= center:
                            code |= (1 << k)
                lbp[i, j] = code
        
        # 统计直方图
        hist, _ = np.histogram(lbp.ravel(), bins=32, range=(0, 256))
        hist = hist / (hist.sum() + 1e-8)
        
        return hist.tolist()
    
    def extract_color_features(self, patch):
        """颜色特征"""
        if len(patch.shape) == 2:
            return [0, 0, 0, 0, 0, 0]
        
        # RGB统计
        features = []
        for c in range(3):
            channel = patch[:, :, c]
            features.append(np.mean(channel))
            features.append(np.std(channel))
        
        # HSV
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        features.append(np.std(hsv[:, :, 0]))  # H标准差
        features.append(np.std(hsv[:, :, 1]))  # S标准差
        
        return features
    
    def extract_all_features(self, patch):
        """提取所有特征"""
        features = []
        
        try:
            features.extend(self.extract_dct_features(patch))
        except:
            features.extend([0] * 5)
        
        try:
            features.extend(self.extract_ela_features(patch))
        except:
            features.extend([0] * 4)
        
        try:
            features.extend(self.extract_noise_features(patch))
        except:
            features.extend([0] * 4)
        
        try:
            features.extend(self.extract_edge_features(patch))
        except:
            features.extend([0] * 4)
        
        try:
            features.extend(self.extract_lbp_features(patch))
        except:
            features.extend([0] * 32)
        
        try:
            features.extend(self.extract_color_features(patch))
        except:
            features.extend([0] * 8)
        
        return np.array(features, dtype=np.float32)


# ============================================================================
# 数据处理
# ============================================================================

class DatasetBuilder:
    """构建像素级训练数据集"""
    
    def __init__(self, feature_extractor, sample_ratio=0.1):
        self.extractor = feature_extractor
        self.sample_ratio = sample_ratio  # 正常像素采样比例
    
    def extract_from_image(self, image_path, mask_path=None):
        """从单张图片提取像素特征和标签"""
        image = load_image(image_path)
        h, w = image.shape[:2]
        
        # 加载mask
        if mask_path and os.path.exists(mask_path):
            mask = load_mask(mask_path)
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h))
            labels = (mask > 0).astype(np.uint8)
        else:
            labels = None
        
        # 收集特征和标签
        features_list = []
        positions = []
        labels_list = []
        
        window = self.extractor.window_size
        stride = self.extractor.stride
        half_window = window // 2
        
        for y in range(half_window, h - half_window, stride):
            for x in range(half_window, w - half_window, stride):
                # 提取patch
                patch = image[y-half_window:y+half_window, x-half_window:x+half_window]
                
                if patch.shape[0] != window or patch.shape[1] != window:
                    continue
                
                # 提取特征
                feat = self.extractor.extract_all_features(patch)
                features_list.append(feat)
                positions.append((y, x))
                
                # 获取标签
                if labels is not None:
                    center_label = labels[y, x]
                    labels_list.append(center_label)
        
        return np.array(features_list), positions, np.array(labels_list) if labels_list else None
    
    def build_dataset(self, data_dirs, output_path=None):
        """构建完整数据集"""
        all_features = []
        all_labels = []
        
        # 统计信息
        total_tampered = 0
        total_normal = 0
        
        for data_dir in data_dirs:
            images_dir = os.path.join(data_dir, 'images')
            masks_dir = os.path.join(data_dir, 'masks')
            
            if not os.path.exists(images_dir):
                continue
            
            image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
            
            print(f"\n处理 {data_dir}: {len(image_files)} 张图片")
            
            for img_file in progress(image_files):
                image_path = os.path.join(images_dir, img_file)
                
                # 查找对应的mask
                mask_name = os.path.splitext(img_file)[0] + '.png'
                mask_path = os.path.join(masks_dir, mask_name)
                if not os.path.exists(mask_path):
                    mask_path = None
                
                # 提取特征
                features, positions, labels = self.extract_from_image(image_path, mask_path)
                
                if labels is None or len(features) == 0:
                    continue
                
                # 分离篡改和正常像素
                tampered_idx = np.where(labels == 1)[0]
                normal_idx = np.where(labels == 0)[0]
                
                # 采样正常像素（处理不平衡）
                if len(normal_idx) > len(tampered_idx) * 5:
                    sample_size = min(len(tampered_idx) * 5, int(len(normal_idx) * self.sample_ratio))
                    sampled_normal = np.random.choice(normal_idx, sample_size, replace=False)
                    selected_idx = np.concatenate([tampered_idx, sampled_normal])
                else:
                    selected_idx = np.arange(len(labels))
                
                all_features.append(features[selected_idx])
                all_labels.append(labels[selected_idx])
                
                total_tampered += len(tampered_idx)
                total_normal += len(normal_idx)
        
        # 合并
        X = np.vstack(all_features)
        y = np.concatenate(all_labels)
        
        print(f"\n数据集统计:")
        print(f"  总样本数: {len(X)}")
        print(f"  篡改像素: {np.sum(y == 1)} ({np.mean(y == 1)*100:.1f}%)")
        print(f"  正常像素: {np.sum(y == 0)} ({np.mean(y == 0)*100:.1f}%)")
        
        # 保存
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            np.savez(output_path, X=X, y=y)
            print(f"  保存到: {output_path}")
        
        return X, y


# ============================================================================
# 模型训练
# ============================================================================

class PixelSegmenter:
    """像素级分割器"""
    
    def __init__(self, model_type='rf'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        
        if model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=10,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=42,
                class_weight='balanced'
            )
        elif model_type == 'gb':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=10,
                learning_rate=0.1,
                random_state=42
            )
    
    def train(self, X, y, val_ratio=0.2):
        """训练模型"""
        print(f"\n训练 {self.model_type.upper()} 模型...")
        
        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 划分
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=val_ratio, random_state=42, stratify=y
        )
        
        print(f"  训练集: {len(X_train)}, 验证集: {len(X_val)}")
        
        # 训练
        self.model.fit(X_train, y_train)
        
        # 评估
        train_score = self.model.score(X_train, y_train)
        val_score = self.model.score(X_val, y_val)
        
        print(f"  训练准确率: {train_score:.4f}")
        print(f"  验证准确率: {val_score:.4f}")
        
        # 详细报告
        y_pred = self.model.predict(X_val)
        print("\n验证集分类报告:")
        print(classification_report(y_val, y_pred, target_names=['正常', '篡改']))
        
        return val_score
    
    def predict_image(self, image, threshold=0.5):
        """预测单张图片"""
        h, w = image.shape[:2]
        
        # 提取特征
        features, positions, _ = self.dataset_builder.extract_from_image(image, None)
        
        if len(features) == 0:
            return np.zeros((h, w), dtype=np.uint8)
        
        # 预测
        features_scaled = self.scaler.transform(features)
        
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(features_scaled)[:, 1]
            pred = (proba > threshold).astype(np.uint8) * 255
        else:
            pred = self.model.predict(features_scaled).astype(np.uint8) * 255
        
        # 重建mask
        mask = np.zeros((h, w), dtype=np.uint8)
        window = self.extractor.window_size
        half_window = window // 2
        
        for i, (y, x) in enumerate(positions):
            mask[y-half_window:y+half_window, x-half_window:x+half_window] = \
                np.maximum(mask[y-half_window:y+half_window, x-half_window:x+half_window], pred[i])
        
        return mask
    
    def save(self, path):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler}, f)
        print(f"模型保存到: {path}")
    
    def load(self, path):
        """加载模型"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.scaler = data['scaler']
        print(f"模型加载自: {path}")


# ============================================================================
# 主流程
# ============================================================================

def main():
    print("=" * 70)
    print("像素级图像篡改分割系统")
    print("=" * 70)
    
    # 配置
    DATA_DIRS = ['tamper_data/easy', 'tamper_data/difficult']
    OUTPUT_DIR = 'results/pixel_segmentation'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 阶段1: 构建数据集
    print("\n" + "=" * 70)
    print("阶段1: 构建像素级数据集")
    print("=" * 70)
    
    extractor = PixelFeatureExtractor(window_size=32, stride=8)
    builder = DatasetBuilder(extractor, sample_ratio=0.1)
    
    dataset_path = os.path.join(OUTPUT_DIR, 'dataset.npz')
    
    if os.path.exists(dataset_path):
        print(f"加载数据集: {dataset_path}")
        data = np.load(dataset_path)
        X, y = data['X'], data['y']
    else:
        X, y = builder.build_dataset(DATA_DIRS, dataset_path)
    
    # 阶段2: 训练模型
    print("\n" + "=" * 70)
    print("阶段2: 训练分割模型")
    print("=" * 70)
    
    segmenter = PixelSegmenter(model_type='rf')
    segmenter.extractor = extractor
    segmenter.dataset_builder = builder
    
    val_score = segmenter.train(X, y)
    
    # 保存模型
    model_path = os.path.join(OUTPUT_DIR, 'model.pkl')
    segmenter.save(model_path)
    
    # 阶段3: 测试评估
    print("\n" + "=" * 70)
    print("阶段3: 测试评估")
    print("=" * 70)
    
    # 在几张图片上测试
    test_dir = 'tamper_data/easy'
    images_dir = os.path.join(test_dir, 'images')
    masks_dir = os.path.join(test_dir, 'masks')
    
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))][:5]
    
    results = []
    
    for img_file in progress(image_files, desc="测试"):
        image_path = os.path.join(images_dir, img_file)
        mask_name = os.path.splitext(img_file)[0] + '.png'
        mask_path = os.path.join(masks_dir, mask_name)
        
        if not os.path.exists(mask_path):
            continue
        
        image = load_image(image_path)
        gt_mask = load_mask(mask_path)
        
        # 预测
        pred_mask = segmenter.predict_image(image, threshold=0.5)
        
        # 后处理
        kernel = np.ones((5, 5), np.uint8)
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
        pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)
        
        # 评估
        pred_binary = (pred_mask > 0).astype(np.uint8)
        gt_binary = (gt_mask > 0).astype(np.uint8)
        
        iou = compute_iou(pred_binary, gt_binary)
        precision, recall = compute_precision_recall(pred_binary, gt_binary)
        f1 = compute_f1(precision, recall)
        
        results.append({
            'image': img_file,
            'iou': float(iou),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        })
        
        # 保存可视化
        vis_output = os.path.join(OUTPUT_DIR, 'visualizations')
        os.makedirs(vis_output, exist_ok=True)
        
        # 叠加显示
        overlay = image.copy()
        overlay[pred_mask > 0] = [0, 255, 0]  # 绿色标记篡改
        vis = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        cv2.imwrite(os.path.join(vis_output, f'{os.path.splitext(img_file)[0]}_pred.png'), vis)
    
    # 统计结果
    if results:
        avg_iou = np.mean([r['iou'] for r in results])
        avg_precision = np.mean([r['precision'] for r in results])
        avg_recall = np.mean([r['recall'] for r in results])
        avg_f1 = np.mean([r['f1'] for r in results])
        
        print(f"\n测试结果 ({len(results)} 张图片):")
        print(f"  IoU: {avg_iou:.4f}")
        print(f"  Precision: {avg_precision:.4f}")
        print(f"  Recall: {avg_recall:.4f}")
        print(f"  F1: {avg_f1:.4f}")
        
        # 保存结果
        final_results = {
            'timestamp': datetime.now().isoformat(),
            'model_type': 'RandomForest',
            'window_size': 32,
            'stride': 8,
            'validation_accuracy': float(val_score),
            'test_results': {
                'iou': float(avg_iou),
                'precision': float(avg_precision),
                'recall': float(avg_recall),
                'f1': float(avg_f1),
                'num_images': len(results)
            },
            'details': results
        }
        
        with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\n结果保存到: {OUTPUT_DIR}/results.json")
    
    print("\n" + "=" * 70)
    print("完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()