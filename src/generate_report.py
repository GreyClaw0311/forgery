"""
实验报告生成脚本
根据测试结果生成Markdown格式的实验报告
"""

import os
import json
from datetime import datetime

def generate_report(results_path, output_path):
    """
    生成实验报告

    Args:
        results_path: 测试结果JSON文件路径
        output_path: 输出报告路径
    """
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    report = []

    # 第一章：背景介绍
    report.append("# 图像篡改检测特征实验报告\n")
    report.append("## 第一章 背景介绍\n")
    report.append(f"**实验时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    report.append("### 1.1 项目背景\n")
    report.append("本项目旨在通过传统OpenCV方法提取图像特征，检测图像是否经过篡改。")
    report.append("图像篡改检测是数字取证领域的重要研究方向，对于验证图像真实性具有重要意义。\n\n")

    report.append("### 1.2 数据集说明\n")
    report.append("| 类别 | 描述 | 样本数量 |\n")
    report.append("|------|------|----------|\n")
    report.append("| easy | 简单篡改图像 | 20张 |\n")
    report.append("| difficult | 复杂篡改图像 | 16张 |\n")
    report.append("| good | 未篡改图像 | 10张 |\n\n")

    report.append("### 1.3 特征介绍\n")
    report.append("本项目实现了10种传统图像篡改检测特征：\n\n")

    feature_descriptions = {
        'ela': '**ELA (Error Level Analysis)**: 错误级别分析，通过重新压缩图像并比较误差来检测篡改区域。',
        'dct': '**DCT域分析**: 离散余弦变换系数分析，篡改区域的DCT系数分布会呈现异常。',
        'cfa': '**CFA插值检测**: Color Filter Array痕迹检测，检测Bayer模式插值痕迹的不一致性。',
        'noise': '**噪声一致性分析**: 传感器噪声模式检测，检测图像中噪声分布的不一致性。',
        'edge': '**边缘一致性检测**: 边缘连续性分析，检测图像边缘的连续性和一致性异常。',
        'lbp': '**LBP纹理特征**: 局部二值模式分析，分析局部纹理模式的一致性。',
        'histogram': '**直方图分析**: 颜色直方图一致性分析，分析颜色直方图分布的一致性。',
        'sift': '**SIFT特征匹配**: 关键点匹配异常检测，检测关键点匹配异常，识别复制粘贴篡改。',
        'fft': '**FFT频域分析**: 傅里叶变换频谱分析，分析图像频谱分布的异常。',
        'metadata': '**元数据分析**: EXIF信息一致性检查，检查图像元数据的完整性和一致性。'
    }

    for i, (name, desc) in enumerate(feature_descriptions.items(), 1):
        report.append(f"{i}. {desc}\n\n")

    # 第二章：实验过程与结果
    report.append("## 第二章 实验过程与结果\n")
    report.append("### 2.1 实验方法\n")
    report.append("对每个特征分别在三类数据上进行测试，记录检测率和篡改分数。\n\n")

    report.append("### 2.2 实验结果\n")

    # 创建结果表格
    report.append("#### 2.2.1 检测率对比表\n")
    report.append("| 特征 | Easy检测率 | Difficult检测率 | Good误报率 | 平均检测率 |\n")
    report.append("|------|------------|-----------------|------------|------------|\n")

    for feature_name, feature_results in results.items():
        easy_rate = feature_results.get('easy', {}).get('detection_rate', 0)
        difficult_rate = feature_results.get('difficult', {}).get('detection_rate', 0)
        good_rate = feature_results.get('good', {}).get('detection_rate', 0)  # 这是误报率
        avg_rate = (easy_rate + difficult_rate + (1 - good_rate)) / 3

        report.append(f"| {feature_name.upper()} | {easy_rate:.2%} | {difficult_rate:.2%} | {good_rate:.2%} | {avg_rate:.2%} |\n")

    report.append("\n#### 2.2.2 平均篡改分数对比表\n")
    report.append("| 特征 | Easy分数 | Difficult分数 | Good分数 |\n")
    report.append("|------|----------|---------------|----------|\n")

    for feature_name, feature_results in results.items():
        easy_score = feature_results.get('easy', {}).get('mean_score', 0)
        difficult_score = feature_results.get('difficult', {}).get('mean_score', 0)
        good_score = feature_results.get('good', {}).get('mean_score', 0)

        report.append(f"| {feature_name.upper()} | {easy_score:.4f} | {difficult_score:.4f} | {good_score:.4f} |\n")

    report.append("\n### 2.3 详细结果分析\n")

    for feature_name, feature_results in results.items():
        report.append(f"\n#### {feature_name.upper()} 特征\n")
        for category, result in feature_results.items():
            report.append(f"- **{category}**: 检测率 {result['detection_rate']:.2%}, ")
            report.append(f"平均分数 {result['mean_score']:.4f} ± {result['std_score']:.4f}, ")
            report.append(f"样本数 {result['total']}\n")

    # 第三章：结论
    report.append("\n## 第三章 结论\n")
    report.append("### 3.1 特征效果分析\n")

    # 计算综合评分
    feature_scores = {}
    for feature_name, feature_results in results.items():
        easy_rate = feature_results.get('easy', {}).get('detection_rate', 0)
        difficult_rate = feature_results.get('difficult', {}).get('detection_rate', 0)
        good_rate = feature_results.get('good', {}).get('detection_rate', 0)
        # 综合评分 = (easy检测率 + difficult检测率) / 2 * (1 - good误报率)
        composite_score = (easy_rate + difficult_rate) / 2 * (1 - good_rate)
        feature_scores[feature_name] = composite_score

    # 排序
    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)

    report.append("根据综合评分（检测率 * (1 - 误报率)）排序：\n\n")
    report.append("| 排名 | 特征 | 综合评分 |\n")
    report.append("|------|------|----------|\n")
    for i, (name, score) in enumerate(sorted_features, 1):
        report.append(f"| {i} | {name.upper()} | {score:.4f} |\n")

    report.append("\n### 3.2 推荐特征组合\n")

    # 选择效果最好的特征
    top_features = [f[0] for f in sorted_features[:5]]
    report.append(f"基于实验结果，推荐使用以下特征组合构建Pipeline：\n\n")
    for f in top_features:
        report.append(f"- {f.upper()}\n")

    report.append("\n### 3.3 后续工作\n")
    report.append("1. 调整各特征的阈值以优化检测效果\n")
    report.append("2. 设计特征融合策略（投票/加权）\n")
    report.append("3. 在更多数据上验证Pipeline效果\n")

    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(''.join(report))

    print(f"报告已生成: {output_path}")

if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from src.config import RESULTS_DIR, REPORTS_DIR

    results_path = os.path.join(RESULTS_DIR, 'feature_test_results.json')
    output_path = os.path.join(REPORTS_DIR, 'feature_experiment_report.md')

    if os.path.exists(results_path):
        generate_report(results_path, output_path)
    else:
        print(f"结果文件不存在: {results_path}")
        print("请先运行 test_features.py")
