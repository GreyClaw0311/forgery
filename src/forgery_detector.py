"""
最终Pipeline - 图像篡改检测
封装特征提取 + 模型预测为统一接口
"""

import os
import sys
import pickle
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from config import RESULTS_DIR

# 导入所有特征模块
from features import (
    feature_ela, feature_dct, feature_cfa, feature_noise,
    feature_edge, feature_lbp, feature_histogram, feature_sift,
    feature_fft, feature_metadata,
    feature_hog, feature_color, feature_adjacency, feature_wavelet,
    feature_gradient, feature_block_dct, feature_jpeg_ghost, feature_local_noise,
    feature_resampling, feature_contrast, feature_blur, feature_saturation,
    feature_splicing
)

try:
    from features import feature_jpeg_block
    HAS_JPEG_BLOCK = True
except ImportError:
    HAS_JPEG_BLOCK = False

# 特征列表
FEATURE_MODULES = [
    ('ela', feature_ela),
    ('dct', feature_dct),
    ('cfa', feature_cfa),
    ('noise', feature_noise),
    ('edge', feature_edge),
    ('lbp', feature_lbp),
    ('histogram', feature_histogram),
    ('sift', feature_sift),
    ('fft', feature_fft),
    ('metadata', feature_metadata),
    ('hog', feature_hog),
    ('color', feature_color),
    ('adjacency', feature_adjacency),
    ('wavelet', feature_wavelet),
    ('gradient', feature_gradient),
    ('block_dct', feature_block_dct),
    ('jpeg_ghost', feature_jpeg_ghost),
    ('local_noise', feature_local_noise),
    ('resampling', feature_resampling),
    ('contrast', feature_contrast),
    ('blur', feature_blur),
    ('saturation', feature_saturation),
    ('splicing', feature_splicing),
]

if HAS_JPEG_BLOCK:
    from features import feature_jpeg_block
    FEATURE_MODULES.append(('jpeg_block', feature_jpeg_block))

FEATURE_NAMES = [f[0] for f in FEATURE_MODULES]


class ForgeryDetector:
    """图像篡改检测器"""
    
    def __init__(self, model_path=None, scaler_path=None):
        """
        初始化检测器
        
        Args:
            model_path: 模型文件路径
            scaler_path: 标准化器文件路径
        """
        self.model = None
        self.scaler = None
        self.feature_names = FEATURE_NAMES
        
        # 加载模型
        if model_path and os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"模型已加载: {model_path}")
        
        # 加载标准化器
        if scaler_path and os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"标准化器已加载: {scaler_path}")
    
    def extract_features(self, image_path):
        """
        提取图像特征
        
        Args:
            image_path: 图像路径
            
        Returns:
            features: 特征向量 (1D numpy array)
        """
        features = []
        
        for feature_name, feature_module in FEATURE_MODULES:
            detect_func = getattr(feature_module, f'detect_tampering_{feature_name}', None)
            if detect_func:
                try:
                    _, score = detect_func(image_path)
                    features.append(float(score))
                except:
                    features.append(0.0)
            else:
                features.append(0.0)
        
        return np.array(features)
    
    def preprocess(self, features):
        """
        特征预处理
        
        Args:
            features: 原始特征向量
            
        Returns:
            processed_features: 处理后的特征向量
        """
        # 1. 处理异常值（log变换）
        processed = features.copy()
        for i in range(len(processed)):
            if processed[i] > 1e6:  # 极大值
                processed[i] = np.log1p(processed[i])
        
        # 2. 标准化
        if self.scaler is not None:
            processed = self.scaler.transform(processed.reshape(1, -1)).flatten()
        
        return processed
    
    def predict(self, image_path):
        """
        预测图像是否被篡改
        
        Args:
            image_path: 图像路径
            
        Returns:
            result: dict with 'is_tampered', 'confidence', 'features'
        """
        # 提取特征
        features = self.extract_features(image_path)
        
        # 预处理
        processed_features = self.preprocess(features)
        
        # 预测
        if self.model is not None:
            prediction = self.model.predict(processed_features.reshape(1, -1))[0]
            
            # 获取概率（如果模型支持）
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(processed_features.reshape(1, -1))[0]
                confidence = float(max(proba))
            else:
                confidence = 1.0
            
            is_tampered = bool(prediction == 1)
        else:
            is_tampered = False
            confidence = 0.0
        
        return {
            'is_tampered': is_tampered,
            'confidence': confidence,
            'features': dict(zip(self.feature_names, features.tolist()))
        }
    
    def train(self, X, y, model_type='rf'):
        """
        训练模型
        
        Args:
            X: 特征矩阵 (N x D)
            y: 标签向量 (N,)
            model_type: 模型类型 ('rf', 'svm', 'lr')
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import RobustScaler
        
        # 预处理
        X_processed = X.copy()
        for i in range(X_processed.shape[1]):
            col = X_processed[:, i]
            if np.any(col > 1e6):
                X_processed[:, i] = np.log1p(col)
        
        # 标准化
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X_processed)
        
        # 选择模型
        if model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                class_weight='balanced',
                random_state=42
            )
        elif model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                class_weight='balanced',
                probability=True,
                random_state=42
            )
        else:  # lr
            self.model = LogisticRegression(
                C=1.0,
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            )
        
        # 训练
        self.model.fit(X_scaled, y)
        print(f"模型训练完成: {model_type}")
        
        return self.model
    
    def save(self, model_path, scaler_path):
        """保存模型和标准化器"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"模型已保存: {model_path}")
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"标准化器已保存: {scaler_path}")


def train_and_save_model():
    """训练并保存模型"""
    import pandas as pd
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("=" * 60)
    print("训练最终模型")
    print("=" * 60)
    
    # 加载特征矩阵
    csv_path = os.path.join(RESULTS_DIR, 'feature_matrix.csv')
    df = pd.read_csv(csv_path)
    
    feature_cols = [col for col in df.columns if col not in ['label', 'category', 'filename']]
    X = df[feature_cols].values
    y = df['label'].values
    
    print(f"\n数据集: {X.shape[0]} 样本, {X.shape[1]} 特征")
    print(f"标签: 篡改={sum(y==1)}, 正常={sum(y==0)}")
    
    # 创建检测器
    detector = ForgeryDetector()
    
    # 训练Random Forest（最佳模型）
    detector.train(X, y, model_type='rf')
    
    # 交叉验证评估
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X_processed = X.copy()
    for i in range(X_processed.shape[1]):
        if np.any(X_processed[:, i] > 1e6):
            X_processed[:, i] = np.log1p(X_processed[:, i])
    
    X_scaled = detector.scaler.transform(X_processed)
    cv_scores = cross_val_score(detector.model, X_scaled, y, cv=cv, scoring='f1_weighted')
    
    print(f"\n交叉验证 F1-score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # 保存模型
    model_path = os.path.join(RESULTS_DIR, 'forgery_model.pkl')
    scaler_path = os.path.join(RESULTS_DIR, 'scaler.pkl')
    detector.save(model_path, scaler_path)
    
    # 测试预测
    print("\n" + "=" * 60)
    print("测试预测")
    print("=" * 60)
    
    # 测试一个篡改图像
    test_tampered = '/tmp/forgery/tamper_data/easy/images/0037_3.jpg'
    if os.path.exists(test_tampered):
        result = detector.predict(test_tampered)
        print(f"\n测试图像: {test_tampered}")
        print(f"预测结果: {'篡改' if result['is_tampered'] else '正常'}")
        print(f"置信度: {result['confidence']:.4f}")
    
    # 测试一个正常图像
    test_normal = '/tmp/forgery/tamper_data/good/0037.jpg'
    if os.path.exists(test_normal):
        result = detector.predict(test_normal)
        print(f"\n测试图像: {test_normal}")
        print(f"预测结果: {'篡改' if result['is_tampered'] else '正常'}")
        print(f"置信度: {result['confidence']:.4f}")
    
    return detector


if __name__ == '__main__':
    detector = train_and_save_model()