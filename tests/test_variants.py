"""
变体特征测试脚本
测试14个高级变体特征在三类数据上的效果
"""

import os
import sys
import json
import time
from datetime import datetime

# 添加src到路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from config import DATA_DIR, RESULTS_DIR, REPORTS_DIR

# 导入变体特征模块
import features.feature_hog as feature_hog
import features.feature_color as feature_color
import features.feature_adjacency as feature_adjacency
import features.feature_wavelet as feature_wavelet
import features.feature_gradient as feature_gradient
import features.feature_block_dct as feature_block_dct
import features.feature_jpeg_ghost as feature_jpeg_ghost
import features.feature_local_noise as feature_local_noise
import features.feature_resampling as feature_resampling
import features.feature_contrast as feature_contrast
import features.feature_blur as feature_blur
import features.feature_saturation as feature_saturation
import features.feature_splicing as feature_splicing
import features.feature_jpeg_block as feature_jpeg_block

# 变体特征列表
VARIANT_FEATURES = [
    'hog', 'color', 'adjacency', 'wavelet', 'gradient',
    'block_dct', 'jpeg_ghost', 'local_noise', 'resampling',
    'contrast', 'blur', 'saturation', 'splicing', 'jpeg_block'
]

# 特征模块映射
VARIANT_MODULES = {
    'hog': feature_hog,
    'color': feature_color,
    'adjacency': feature_adjacency,
    'wavelet': feature_wavelet,
    'gradient': feature_gradient,
    'block_dct': feature_block_dct,
    'jpeg_ghost': feature_jpeg_ghost,
    'local_noise': feature_local_noise,
    'resampling': feature_resampling,
    'contrast': feature_contrast,
    'blur': feature_blur,
    'saturation': feature_saturation,
    'splicing': feature_splicing,
    'jpeg_block': feature_jpeg_block
}

def get_variant_module(feature_name):
    """获取变体特征模块"""
    return VARIANT_MODULES.get(feature_name)

def test_single_variant(feature_name, category):
    """
    测试单个变体特征在特定类别数据上的效果

    Args:
        feature_name: 特征名称
        category: 数据类别 (easy, difficult, good)

    Returns:
        results: 测试结果字典
    """
    module = get_variant_module(feature_name)
    if module is None:
        return None

    detect_func = getattr(module, f'detect_tampering_{feature_name}', None)
    if detect_func is None:
        return None

    # 获取图像路径
    if category == 'good':
        img_dir = os.path.join(DATA_DIR, category)
        img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
    else:
        img_dir = os.path.join(DATA_DIR, category, 'images')
        img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]

    if not os.path.exists(img_dir):
        return None

    results = {
        'feature': feature_name,
        'category': category,
        'total': 0,
        'detected': 0,
        'scores': [],
        'errors': 0
    }

    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        try:
            is_tampered, score = detect_func(img_path)
            results['total'] += 1
            results['scores'].append(float(score))
            if is_tampered:
                results['detected'] += 1
        except Exception as e:
            results['errors'] += 1
            print(f"  错误 [{img_file}]: {str(e)[:50]}")

    # 计算统计指标
    if results['total'] > 0:
        results['detection_rate'] = results['detected'] / results['total']
        results['mean_score'] = sum(results['scores']) / len(results['scores']) if results['scores'] else 0
        results['std_score'] = (sum((s - results['mean_score'])**2 for s in results['scores']) / len(results['scores'])) ** 0.5 if results['scores'] else 0
    else:
        results['detection_rate'] = 0
        results['mean_score'] = 0
        results['std_score'] = 0

    return results

def run_all_variant_tests():
    """运行所有变体特征测试"""
    print("=" * 60)
    print("图像篡改检测 - 变体特征实验")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    all_results = {}

    for feature_name in VARIANT_FEATURES:
        print(f"\n测试特征: {feature_name.upper()}")
        all_results[feature_name] = {}

        for category in ['easy', 'difficult', 'good']:
            print(f"  类别: {category}...", end=" ", flush=True)
            result = test_single_variant(feature_name, category)
            if result:
                all_results[feature_name][category] = result
                print(f"检测率: {result['detection_rate']:.2%}, 平均分: {result['mean_score']:.4f}")
            else:
                print("跳过")

    # 保存结果
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_file = os.path.join(RESULTS_DIR, 'variant_test_results.json')
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到: {result_file}")
    return all_results

def generate_report(results):
    """生成测试报告"""
    report_lines = []
    report_lines.append("# 图像篡改检测变体特征实验报告")
    report_lines.append("## 第一章 背景介绍")
    report_lines.append(f"**实验时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append("### 1.1 变体特征介绍")
    report_lines.append("本项目在10个基础特征的基础上，实现了14个高级变体特征：")
    report_lines.append("")
    
    feature_descriptions = {
        'hog': 'HOG (Histogram of Oriented Gradients): 方向梯度直方图，分析图像局部梯度方向分布的一致性',
        'color': 'COLOR (颜色一致性): 分析图像颜色分布的空间一致性',
        'adjacency': 'ADJACENCY (邻域一致性): 分析像素邻域关系的连续性',
        'wavelet': 'WAVELET (小波分析): 使用小波变换分析图像的多尺度特征',
        'gradient': 'GRADIENT (梯度一致性): 分析图像梯度的空间分布一致性',
        'block_dct': 'BLOCK_DCT (分块DCT): 分块分析DCT系数的分布一致性',
        'jpeg_ghost': 'JPEG_GHOST (JPEG Ghost): 检测JPEG压缩伪影的异常区域',
        'local_noise': 'LOCAL_NOISE (局部噪声): 分析图像局部噪声分布的一致性',
        'resampling': 'RESAMPLING (重采样检测): 检测图像中的插值痕迹',
        'contrast': 'CONTRAST (对比度一致性): 分析图像局部对比度的一致性',
        'blur': 'BLUR (模糊检测): 检测图像局部模糊程度的一致性',
        'saturation': 'SATURATION (饱和度一致性): 分析图像颜色饱和度的空间一致性',
        'splicing': 'SPLICING (拼接检测): 检测图像中的拼接边缘和区域',
        'jpeg_block': 'JPEG_BLOCK (JPEG块效应): 检测JPEG压缩的8x8块边界伪影'
    }
    
    for i, (feat, desc) in enumerate(feature_descriptions.items(), 1):
        report_lines.append(f"{i}. **{feat.upper()}**: {desc}")
    
    report_lines.append("")
    report_lines.append("### 1.2 数据集说明")
    report_lines.append("| 类别 | 描述 | 样本数量 |")
    report_lines.append("|------|------|----------|")
    report_lines.append("| easy | 简单篡改图像 | 20张 |")
    report_lines.append("| difficult | 复杂篡改图像 | 16张 |")
    report_lines.append("| good | 未篡改图像 | 10张 |")
    report_lines.append("")
    
    report_lines.append("## 第二章 实验过程与结果")
    report_lines.append("### 2.1 实验方法")
    report_lines.append("对每个变体特征分别在三类数据上进行测试，记录检测率和篡改分数。")
    report_lines.append("")
    
    report_lines.append("### 2.2 实验结果")
    report_lines.append("#### 2.2.1 检测率对比表")
    report_lines.append("| 特征 | Easy检测率 | Difficult检测率 | Good误报率 | 平均检测率 |")
    report_lines.append("|------|------------|-----------------|------------|------------|")
    
    summary_data = []
    
    for feat_name in VARIANT_FEATURES:
        if feat_name in results:
            easy_data = results[feat_name].get('easy', {})
            difficult_data = results[feat_name].get('difficult', {})
            good_data = results[feat_name].get('good', {})
            
            easy_rate = easy_data.get('detection_rate', 0) * 100
            difficult_rate = difficult_data.get('detection_rate', 0) * 100
            good_rate = good_data.get('detection_rate', 0) * 100  # 误报率
            
            avg_rate = (easy_rate + difficult_rate + (100 - good_rate)) / 3
            
            report_lines.append(f"| {feat_name.upper()} | {easy_rate:.2f}% | {difficult_rate:.2f}% | {good_rate:.2f}% | {avg_rate:.2f}% |")
            
            summary_data.append({
                'feature': feat_name,
                'easy_rate': easy_rate,
                'difficult_rate': difficult_rate,
                'good_fp_rate': good_rate,
                'avg_rate': avg_rate
            })
    
    report_lines.append("")
    report_lines.append("#### 2.2.2 平均篡改分数对比表")
    report_lines.append("| 特征 | Easy分数 | Difficult分数 | Good分数 |")
    report_lines.append("|------|----------|---------------|----------|")
    
    for feat_name in VARIANT_FEATURES:
        if feat_name in results:
            easy_data = results[feat_name].get('easy', {})
            difficult_data = results[feat_name].get('difficult', {})
            good_data = results[feat_name].get('good', {})
            
            easy_score = easy_data.get('mean_score', 0)
            difficult_score = difficult_data.get('mean_score', 0)
            good_score = good_data.get('mean_score', 0)
            
            report_lines.append(f"| {feat_name.upper()} | {easy_score:.4f} | {difficult_score:.4f} | {good_score:.4f} |")
    
    report_lines.append("")
    report_lines.append("### 2.3 详细结果分析")
    
    for feat_name in VARIANT_FEATURES:
        if feat_name in results:
            report_lines.append(f"\n#### {feat_name.upper()} 特征")
            for category in ['easy', 'difficult', 'good']:
                if category in results[feat_name]:
                    data = results[feat_name][category]
                    report_lines.append(f"- **{category}**: 检测率 {data['detection_rate']:.2%}, 平均分数 {data['mean_score']:.4f} ± {data['std_score']:.4f}, 样本数 {data['total']}")
    
    report_lines.append("")
    report_lines.append("## 第三章 结论")
    report_lines.append("### 3.1 特征效果分析")
    
    # 计算综合评分
    for item in summary_data:
        detection_rate = (item['easy_rate'] + item['difficult_rate']) / 2
        false_positive_rate = item['good_fp_rate']
        item['composite_score'] = detection_rate * (1 - false_positive_rate / 100) / 100
    
    summary_data.sort(key=lambda x: x['composite_score'], reverse=True)
    
    report_lines.append("根据综合评分（检测率 * (1 - 误报率)）排序：")
    report_lines.append("")
    report_lines.append("| 排名 | 特征 | 综合评分 |")
    report_lines.append("|------|------|----------|")
    
    for i, item in enumerate(summary_data, 1):
        report_lines.append(f"| {i} | {item['feature'].upper()} | {item['composite_score']:.4f} |")
    
    report_lines.append("")
    report_lines.append("### 3.2 推荐变体特征组合")
    
    # 选择综合评分最高的特征
    top_features = [item['feature'] for item in summary_data[:5] if item['composite_score'] > 0]
    
    if top_features:
        report_lines.append("基于实验结果，推荐使用以下变体特征：")
        for feat in top_features:
            report_lines.append(f"- {feat.upper()}")
    else:
        report_lines.append("当前阈值下所有变体特征的综合评分较低，建议调整阈值后重新测试。")
    
    report_lines.append("")
    report_lines.append("### 3.3 后续工作")
    report_lines.append("1. 调整各变体特征的阈值以优化检测效果")
    report_lines.append("2. 研究变体特征与基础特征的互补性")
    report_lines.append("3. 设计特征融合策略")
    report_lines.append("4. 在更多数据上验证Pipeline效果")
    
    return '\n'.join(report_lines)

if __name__ == '__main__':
    results = run_all_variant_tests()
    
    # 生成报告
    report = generate_report(results)
    
    # 保存报告
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_file = os.path.join(REPORTS_DIR, 'variant_experiment_report.md')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n报告已保存到: {report_file}")