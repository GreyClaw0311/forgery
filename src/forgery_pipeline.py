"""
步骤3.5.5 + 3.6: 最终模型保存 + Pipeline开发
完整的图像篡改检测Pipeline

输入: 图片路径
处理: 自动提取特征 → 标准化 → 模型预测
输出: 是否篡改 + 置信度
"""

import os
import sys
import pickle
import numpy as np

sys.path.insert(0, '/tmp/forgery/src')

# Top 10 特征
FEATURE_NAMES = ['jpeg_block', 'contrast', 'saturation', 'jpeg_ghost', 'fft', 
                 'cfa', 'edge', 'color', 'resampling', 'splicing']

# 最优阈值
OPTIMAL_THRESHOLD = 0.85


class ForgeryDetectionPipeline:
    """图像篡改检测Pipeline"""
    
    def __init__(self, model_path=None, scaler_path=None, threshold=None):
        """
        初始化Pipeline
        
        Args:
            model_path: 模型文件路径
            scaler_path: 标准化器文件路径
            threshold: 分类阈值
        """
        self.model = None
        self.scaler = None
        self.threshold = threshold or OPTIMAL_THRESHOLD
        self.feature_names = FEATURE_NAMES
        
        # 默认路径
        if model_path is None:
            model_path = '/tmp/forgery/results/full/optimized_model.pkl'
        if scaler_path is None:
            scaler_path = '/tmp/forgery/results/full/optimized_scaler.pkl'
        
        # 加载模型
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"模型已加载: {model_path}")
        
        # 加载标准化器
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"标准化器已加载: {scaler_path}")
    
    def extract_features(self, image_path):
        """
        提取图片特征
        
        Args:
            image_path: 图片路径
            
        Returns:
            features: 特征向量
        """
        features = []
        
        for fname in self.feature_names:
            try:
                module = __import__(f'features.feature_{fname}', fromlist=[''])
                detect_func = getattr(module, f'detect_tampering_{fname}', None)
                if detect_func:
                    _, score = detect_func(image_path)
                    features.append(float(score))
                else:
                    features.append(0.0)
            except Exception as e:
                print(f"警告: {fname} 提取失败: {e}")
                features.append(0.0)
        
        return np.array(features)
    
    def predict(self, image_path):
        """
        预测图片是否被篡改
        
        Args:
            image_path: 图片路径
            
        Returns:
            result: dict
                - is_tampered: bool, 是否篡改
                - confidence: float, 置信度
                - probability: float, 篡改概率
                - features: dict, 特征值
        """
        # 提取特征
        features = self.extract_features(image_path)
        
        # 标准化
        if self.scaler is not None:
            features_scaled = self.scaler.transform(features.reshape(1, -1))
        else:
            features_scaled = features.reshape(1, -1)
        
        # 预测
        if self.model is not None:
            # 获取概率
            probability = self.model.predict_proba(features_scaled)[0, 1]
            
            # 使用阈值判断
            is_tampered = probability >= self.threshold
            
            # 置信度
            confidence = max(probability, 1 - probability)
        else:
            is_tampered = False
            probability = 0.0
            confidence = 0.0
        
        return {
            'is_tampered': bool(is_tampered),
            'confidence': float(confidence),
            'probability': float(probability),
            'threshold': float(self.threshold),
            'features': dict(zip(self.feature_names, features.tolist()))
        }
    
    def predict_batch(self, image_paths):
        """
        批量预测
        
        Args:
            image_paths: 图片路径列表
            
        Returns:
            results: list of dict
        """
        results = []
        for path in image_paths:
            result = self.predict(path)
            result['image_path'] = path
            results.append(result)
        return results


def test_pipeline():
    """测试Pipeline"""
    print("=" * 60)
    print("测试图像篡改检测Pipeline")
    print("=" * 60)
    
    # 初始化Pipeline
    pipeline = ForgeryDetectionPipeline()
    
    # 测试图片
    test_images = [
        ('/tmp/forgery/tamper_data_full/processed/easy/images/doctamper-fcd_0.jpg', '篡改'),
        ('/tmp/forgery/tamper_data_full/processed/difficult/images/rtm_cover_0001.jpg', '篡改'),
        ('/tmp/forgery/tamper_data_full/processed/good/rtm_good_0001.jpg', '正常'),
    ]
    
    print("\n测试结果:")
    print("-" * 80)
    
    correct = 0
    total = 0
    
    for img_path, true_label in test_images:
        if not os.path.exists(img_path):
            print(f"文件不存在: {img_path}")
            continue
        
        result = pipeline.predict(img_path)
        
        pred_label = '篡改' if result['is_tampered'] else '正常'
        is_correct = pred_label == true_label
        
        if is_correct:
            correct += 1
        total += 1
        
        status = "✅" if is_correct else "❌"
        
        print(f"\n{status} {os.path.basename(img_path)}")
        print(f"  真实标签: {true_label}")
        print(f"  预测结果: {pred_label}")
        print(f"  篡改概率: {result['probability']:.4f}")
        print(f"  置信度: {result['confidence']:.4f}")
        print(f"  阈值: {result['threshold']}")
    
    print("\n" + "=" * 80)
    print(f"测试准确率: {correct}/{total} ({correct/total*100:.1f}%)")
    print("=" * 80)
    
    return pipeline


def main():
    """主函数"""
    print("=" * 60)
    print("步骤3.5.5 + 3.6: Pipeline开发")
    print("=" * 60)
    
    # 测试Pipeline
    pipeline = test_pipeline()
    
    # 保存Pipeline
    import json
    from datetime import datetime
    
    pipeline_info = {
        'feature_names': FEATURE_NAMES,
        'optimal_threshold': OPTIMAL_THRESHOLD,
        'model_file': 'optimized_model.pkl',
        'scaler_file': 'optimized_scaler.pkl',
        'created_at': datetime.now().isoformat(),
        'performance': {
            'fpr': 0.3684,
            'fnr': 0.1138,
            'precision': 0.9787,
            'recall': 0.8862,
            'f1': 0.9302
        }
    }
    
    output_dir = '/tmp/forgery/results/full'
    with open(os.path.join(output_dir, 'pipeline_info.json'), 'w') as f:
        json.dump(pipeline_info, f, indent=2)
    
    print(f"\nPipeline信息已保存: {output_dir}/pipeline_info.json")
    
    return pipeline


if __name__ == '__main__':
    pipeline = main()