#!/usr/bin/env python3
"""
图像篡改检测服务测试脚本

功能:
1. 单张图片测试
2. 批量图片测试
3. 生成测试报告 (效果 + 性能)
4. 保存推理结果图片

数据集结构:
|-- test
|   |-- images
|   `-- masks

判断逻辑:
- mask 全黑 → good (未篡改)
- mask 非全黑 → 篡改

使用方法:
    # 单张测试
    python test_service.py --mode single --image test.jpg --algorithm ml
    
    # 批量测试
    python test_service.py --mode batch --data_dir ./test --algorithm ml
    
    # 全算法对比测试
    python test_service.py --mode batch --data_dir ./test --algorithm all
"""

import os
import sys
import json
import base64
import argparse
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import requests
import cv2
import numpy as np


class TamperDetectionTester:
    """篡改检测服务测试器"""
    
    def __init__(self, server_url: str = "http://localhost:8000", output_dir: str = "./test_results"):
        """
        初始化测试器
        
        Args:
            server_url: 服务地址
            output_dir: 结果输出目录
        """
        self.server_url = server_url
        self.output_dir = output_dir
        self.timeout = 120  # 请求超时时间
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    
    def check_service(self) -> bool:
        """检查服务是否正常运行"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✓ 服务正常运行")
                print(f"  状态: {data.get('status')}")
                print(f"  算法: {data.get('algorithms')}")
                print(f"  GB分类器: {'已加载' if data.get('gb_classifier_loaded') else '未加载'}")
                return True
            return False
        except Exception as e:
            print(f"✗ 服务连接失败: {e}")
            return False
    
    def detect_single(self, image_path: str, algorithm: str = "ml", skip_gb: bool = False) -> Dict:
        """
        单张图片检测
        
        Args:
            image_path: 图片路径
            algorithm: 算法名称
            skip_gb: 是否跳过 GB 前置检测
            
        Returns:
            检测结果
        """
        # 读取图片并编码为 Base64
        with open(image_path, 'rb') as f:
            image_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        # 发送请求
        payload = {
            'image_base64': image_base64,
            'algorithm': algorithm
        }
        
        if skip_gb:
            payload['skip_gb'] = 'true'
        
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.server_url}/tamper_detection/v1/tamper_detect_img",
                data=payload,
                timeout=self.timeout
            )
            request_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                result['request_time'] = request_time
                return result
            else:
                return {
                    'status': 'error',
                    'message': f"HTTP {response.status_code}: {response.text}",
                    'request_time': request_time
                }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'request_time': time.time() - start_time
            }
    
    def check_mask_label(self, mask_path: str) -> int:
        """
        根据 mask 判断标签
        
        Args:
            mask_path: mask 图片路径
            
        Returns:
            0: 正常 (mask 全黑)
            1: 篡改 (mask 非全黑)
        """
        if not os.path.exists(mask_path):
            # 如果没有对应的 mask，默认认为是篡改图片
            return 1
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return 1
        
        # 判断是否全黑 (所有像素值为 0)
        if mask.sum() == 0:
            return 0  # 正常
        else:
            return 1  # 篡改
    
    def save_result_image(self, image_path: str, result: Dict, output_path: str, 
                          true_label: int = None, mask_path: str = None) -> bool:
        """
        保存检测结果图片
        
        Args:
            image_path: 原图路径
            result: 检测结果
            output_path: 输出路径
            true_label: 真实标签
            mask_path: mask 路径
            
        Returns:
            是否成功
        """
        try:
            # 读取原图
            image = cv2.imread(image_path)
            if image is None:
                return False
            
            # 创建结果图 (2列: 原图+结果, mask+预测)
            h, w = image.shape[:2]
            
            # 创建画布
            canvas = np.ones((h, w * 2 + 20, 3), dtype=np.uint8) * 255
            
            # 放置原图
            canvas[:h, :w] = image
            
            # 中间分隔线
            canvas[:, w:w+20] = 200
            
            # 创建预测结果图
            pred_image = image.copy()
            
            # 绘制篡改区域
            if result.get('tamper_regions'):
                for region in result['tamper_regions']:
                    x1, y1 = region['left_top']
                    x2, y2 = region['right_bottom']
                    cv2.rectangle(pred_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            # 添加预测信息
            pred_label = 1 if result.get('is_tampered') else 0
            pred_text = "TAMPERED" if pred_label else "GOOD"
            color = (0, 0, 255) if pred_label else (0, 255, 0)
            cv2.putText(pred_image, f"Pred: {pred_text}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(pred_image, f"Conf: {result.get('confidence', 0):.3f}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # 添加真实标签
            if true_label is not None:
                true_text = "TAMPERED" if true_label else "GOOD"
                true_color = (0, 100, 255) if true_label else (100, 255, 0)
                cv2.putText(pred_image, f"True: {true_text}", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, true_color, 2)
                
                # 标记是否正确
                is_correct = pred_label == true_label
                correct_text = "CORRECT" if is_correct else "WRONG"
                correct_color = (0, 255, 0) if is_correct else (0, 0, 255)
                cv2.putText(pred_image, correct_text, (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, correct_color, 2)
            
            # 放置预测结果图
            canvas[:h, w+20:] = pred_image
            
            # 保存
            cv2.imwrite(output_path, canvas)
            return True
        except Exception as e:
            print(f"保存图片失败: {e}")
            return False
    
    def test_single(self, image_path: str, algorithm: str = "ml", save_image: bool = True,
                    mask_path: str = None):
        """
        单张图片测试
        
        Args:
            image_path: 图片路径
            algorithm: 算法名称
            save_image: 是否保存结果图片
            mask_path: mask 路径 (用于判断真实标签)
        """
        print("=" * 60)
        print("单张图片测试")
        print("=" * 60)
        
        if not os.path.exists(image_path):
            print(f"✗ 图片不存在: {image_path}")
            return
        
        print(f"图片: {image_path}")
        print(f"算法: {algorithm}")
        print("-" * 60)
        
        # 判断真实标签
        true_label = None
        if mask_path:
            true_label = self.check_mask_label(mask_path)
            print(f"真实标签: {'篡改' if true_label else '正常'}")
        
        # 检测
        start_time = time.time()
        result = self.detect_single(image_path, algorithm)
        total_time = time.time() - start_time
        
        # 打印结果
        print(f"\n检测结果:")
        print(f"  状态: {result.get('status', 'N/A')}")
        print(f"  是否篡改: {result.get('is_tampered', 'N/A')}")
        print(f"  置信度: {result.get('confidence', 0):.4f}")
        print(f"  GB置信度: {result.get('gb_confidence', 'N/A')}")
        print(f"  处理时间: {result.get('processing_time', 0):.3f}s")
        print(f"  请求时间: {result.get('request_time', 0):.3f}s")
        
        if result.get('tamper_regions'):
            print(f"  篡改区域: {len(result['tamper_regions'])} 个")
            for i, region in enumerate(result['tamper_regions']):
                print(f"    区域{i+1}: {region['left_top']} -> {region['right_bottom']}")
        
        if true_label is not None:
            pred_label = 1 if result.get('is_tampered') else 0
            is_correct = pred_label == true_label
            print(f"\n  判断结果: {'✓ 正确' if is_correct else '✗ 错误'}")
        
        # 保存结果图片
        if save_image:
            image_name = os.path.basename(image_path)
            output_name = f"{os.path.splitext(image_name)[0]}_{algorithm}.jpg"
            output_path = os.path.join(self.output_dir, 'images', output_name)
            
            if self.save_result_image(image_path, result, output_path, true_label, mask_path):
                print(f"\n✓ 结果图片已保存: {output_path}")
    
    def test_batch(self, data_dir: str, algorithm: str = "ml", save_images: bool = True) -> Dict:
        """
        批量图片测试
        
        Args:
            data_dir: 数据目录 (包含 images/ 和 masks/ 子目录)
            algorithm: 算法名称
            save_images: 是否保存结果图片
            
        Returns:
            测试报告
        """
        print("=" * 60)
        print("批量图片测试")
        print("=" * 60)
        
        if not os.path.exists(data_dir):
            print(f"✗ 数据目录不存在: {data_dir}")
            return {}
        
        # 收集测试图片
        images_dir = os.path.join(data_dir, 'images')
        masks_dir = os.path.join(data_dir, 'masks')
        
        if not os.path.exists(images_dir):
            print(f"✗ 图片目录不存在: {images_dir}")
            return {}
        
        # 遍历图片
        test_images = []
        for f in sorted(os.listdir(images_dir)):
            if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                image_path = os.path.join(images_dir, f)
                
                # 查找对应的 mask
                mask_name = os.path.splitext(f)[0] + '.png'  # mask 通常是 png
                mask_path = os.path.join(masks_dir, mask_name)
                
                # 如果 mask 不存在，尝试其他扩展名
                if not os.path.exists(mask_path):
                    for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                        mask_path = os.path.join(masks_dir, os.path.splitext(f)[0] + ext)
                        if os.path.exists(mask_path):
                            break
                
                # 根据 mask 判断标签
                true_label = self.check_mask_label(mask_path)
                
                test_images.append({
                    'path': image_path,
                    'mask_path': mask_path if os.path.exists(mask_path) else None,
                    'label': true_label,
                    'category': 'tampered' if true_label == 1 else 'good'
                })
        
        if not test_images:
            print("✗ 未找到测试图片")
            return {}
        
        # 统计
        tampered_count = sum(1 for t in test_images if t['label'] == 1)
        good_count = sum(1 for t in test_images if t['label'] == 0)
        
        print(f"数据目录: {data_dir}")
        print(f"算法: {algorithm}")
        print(f"测试图片数: {len(test_images)}")
        print(f"  篡改图片: {tampered_count}")
        print(f"  正常图片: {good_count}")
        print("-" * 60)
        
        # 测试统计
        results = []
        correct = 0
        total = 0
        total_time = 0
        
        # 分类统计
        tp = 0  # 真阳性 (篡改->篡改)
        tn = 0  # 真阴性 (正常->正常)
        fp = 0  # 假阳性 (正常->篡改)
        fn = 0  # 假阴性 (篡改->正常)
        
        # 置信度统计
        tampered_confidences = []
        normal_confidences = []
        
        # 时间统计
        processing_times = []
        
        for i, item in enumerate(test_images):
            image_path = item['path']
            mask_path = item['mask_path']
            true_label = item['label']
            
            # 检测
            result = self.detect_single(image_path, algorithm)
            
            if result.get('status') == '0001:解析成功.':
                pred_label = 1 if result.get('is_tampered') else 0
                confidence = result.get('confidence', 0)
                processing_time = result.get('processing_time', 0)
                request_time = result.get('request_time', 0)
                gb_confidence = result.get('gb_confidence')
                
                total += 1
                total_time += request_time
                processing_times.append(processing_time)
                
                # 统计
                if pred_label == true_label:
                    correct += 1
                
                if true_label == 1:
                    tampered_confidences.append(confidence)
                    if pred_label == 1:
                        tp += 1
                    else:
                        fn += 1
                else:
                    normal_confidences.append(confidence)
                    if pred_label == 0:
                        tn += 1
                    else:
                        fp += 1
                
                # 保存图片
                if save_images:
                    image_name = os.path.basename(image_path)
                    output_name = f"{os.path.splitext(image_name)[0]}_{algorithm}.jpg"
                    output_path = os.path.join(self.output_dir, 'images', output_name)
                    self.save_result_image(image_path, result, output_path, true_label, mask_path)
                
                results.append({
                    'image': os.path.basename(image_path),
                    'true_label': true_label,
                    'pred_label': pred_label,
                    'correct': pred_label == true_label,
                    'confidence': confidence,
                    'gb_confidence': gb_confidence,
                    'processing_time': processing_time,
                    'request_time': request_time
                })
            else:
                # 检测失败
                results.append({
                    'image': os.path.basename(image_path),
                    'true_label': true_label,
                    'pred_label': None,
                    'correct': False,
                    'confidence': None,
                    'gb_confidence': None,
                    'processing_time': None,
                    'request_time': result.get('request_time', 0),
                    'error': result.get('message', 'Unknown error')
                })
            
            # 进度
            if (i + 1) % 10 == 0 or (i + 1) == len(test_images):
                print(f"  进度: {i+1}/{len(test_images)} ({(i+1)/len(test_images)*100:.1f}%)")
        
        # 计算指标
        accuracy = correct / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        avg_time = total_time / total if total > 0 else 0
        avg_processing_time = np.mean(processing_times) if processing_times else 0
        avg_tampered_conf = np.mean(tampered_confidences) if tampered_confidences else 0
        avg_normal_conf = np.mean(normal_confidences) if normal_confidences else 0
        
        # 生成报告
        report = {
            'test_info': {
                'data_dir': data_dir,
                'algorithm': algorithm,
                'test_time': datetime.now().isoformat(),
                'total_images': len(test_images),
                'tested_images': total,
                'tampered_images': tampered_count,
                'good_images': good_count
            },
            'performance': {
                'total_time': round(total_time, 2),
                'avg_time': round(avg_time, 4),
                'avg_processing_time': round(avg_processing_time, 4),
                'min_time': round(min(processing_times), 4) if processing_times else 0,
                'max_time': round(max(processing_times), 4) if processing_times else 0,
                'p50_time': round(np.percentile(processing_times, 50), 4) if processing_times else 0,
                'p95_time': round(np.percentile(processing_times, 95), 4) if processing_times else 0,
                'fps': round(total / total_time, 2) if total_time > 0 else 0
            },
            'effectiveness': {
                'accuracy': round(accuracy, 4),
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1': round(f1, 4),
                'fpr': round(fpr, 4),
                'fnr': round(fnr, 4),
                'tp': tp,
                'tn': tn,
                'fp': fp,
                'fn': fn
            },
            'confidence_stats': {
                'avg_tampered_confidence': round(avg_tampered_conf, 4),
                'avg_normal_confidence': round(avg_normal_conf, 4),
                'tampered_count': len(tampered_confidences),
                'normal_count': len(normal_confidences)
            },
            'detailed_results': results
        }
        
        # 打印报告
        self._print_report(report)
        
        # 保存报告
        report_path = os.path.join(self.output_dir, f"test_report_{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n✓ 测试报告已保存: {report_path}")
        
        return report
    
    def _print_report(self, report: Dict):
        """打印测试报告"""
        print("\n" + "=" * 60)
        print("测试报告")
        print("=" * 60)
        
        # 基本信息
        info = report['test_info']
        print(f"\n【测试信息】")
        print(f"  数据目录: {info['data_dir']}")
        print(f"  测试算法: {info['algorithm']}")
        print(f"  测试时间: {info['test_time']}")
        print(f"  测试图片: {info['tested_images']}/{info['total_images']}")
        print(f"  篡改图片: {info['tampered_images']}")
        print(f"  正常图片: {info['good_images']}")
        
        # 效果指标
        eff = report['effectiveness']
        print(f"\n【效果指标】")
        print(f"  Accuracy:  {eff['accuracy']:.4f} ({eff['tp']+eff['tn']}/{eff['tp']+eff['tn']+eff['fp']+eff['fn']})")
        print(f"  Precision: {eff['precision']:.4f}")
        print(f"  Recall:    {eff['recall']:.4f}")
        print(f"  F1-score:  {eff['f1']:.4f}")
        print(f"  FPR (假阳性率): {eff['fpr']:.4f}")
        print(f"  FNR (假阴性率): {eff['fnr']:.4f}")
        print(f"\n  混淆矩阵:")
        print(f"              预测篡改  预测正常")
        print(f"  实际篡改      {eff['tp']:4d}      {eff['fn']:4d}")
        print(f"  实际正常      {eff['fp']:4d}      {eff['tn']:4d}")
        
        # 性能指标
        perf = report['performance']
        print(f"\n【性能指标】")
        print(f"  总耗时: {perf['total_time']:.2f}s")
        print(f"  平均处理时间: {perf['avg_processing_time']*1000:.2f}ms")
        print(f"  平均请求时间: {perf['avg_time']*1000:.2f}ms")
        print(f"  最小耗时: {perf['min_time']*1000:.2f}ms")
        print(f"  最大耗时: {perf['max_time']*1000:.2f}ms")
        print(f"  P50耗时: {perf['p50_time']*1000:.2f}ms")
        print(f"  P95耗时: {perf['p95_time']*1000:.2f}ms")
        print(f"  处理速度: {perf['fps']:.2f} FPS")
        
        # 置信度统计
        conf = report['confidence_stats']
        print(f"\n【置信度统计】")
        print(f"  篡改图片平均置信度: {conf['avg_tampered_confidence']:.4f} ({conf['tampered_count']}张)")
        print(f"  正常图片平均置信度: {conf['avg_normal_confidence']:.4f} ({conf['normal_count']}张)")
    
    def compare_algorithms(self, data_dir: str, algorithms: List[str] = None, save_images: bool = False):
        """
        多算法对比测试
        
        Args:
            data_dir: 数据目录
            algorithms: 算法列表
            save_images: 是否保存图片
        """
        if algorithms is None:
            algorithms = ['ela', 'dct', 'fusion', 'ml']
        
        print("=" * 60)
        print("多算法对比测试")
        print("=" * 60)
        
        reports = {}
        for algo in algorithms:
            print(f"\n>>> 测试算法: {algo}")
            print("-" * 60)
            report = self.test_batch(data_dir, algo, save_images)
            reports[algo] = report
        
        # 打印对比表
        print("\n" + "=" * 60)
        print("算法对比结果")
        print("=" * 60)
        print(f"\n{'算法':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Avg Time(ms)':<15}")
        print("-" * 70)
        
        for algo, report in reports.items():
            if report:
                eff = report['effectiveness']
                perf = report['performance']
                print(f"{algo:<10} {eff['accuracy']:<10.4f} {eff['precision']:<10.4f} "
                      f"{eff['recall']:<10.4f} {eff['f1']:<10.4f} {perf['avg_processing_time']*1000:<15.2f}")
        
        # 保存对比报告
        compare_path = os.path.join(self.output_dir, f"compare_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(compare_path, 'w', encoding='utf-8') as f:
            json.dump(reports, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n✓ 对比报告已保存: {compare_path}")


def main():
    parser = argparse.ArgumentParser(description='图像篡改检测服务测试脚本')
    parser.add_argument('--mode', type=str, choices=['single', 'batch', 'compare'], default='single',
                       help='测试模式: single(单张), batch(批量), compare(对比)')
    parser.add_argument('--image', type=str, help='单张测试图片路径')
    parser.add_argument('--mask', type=str, help='单张测试对应的mask路径')
    parser.add_argument('--data_dir', type=str, default='./test',
                       help='批量测试数据目录 (包含 images/ 和 masks/)')
    parser.add_argument('--algorithm', type=str, default='ml',
                       help='算法名称: ela/dct/fusion/ml/all')
    parser.add_argument('--server', type=str, default='http://localhost:8000',
                       help='服务地址')
    parser.add_argument('--output', type=str, default='./test_results',
                       help='结果输出目录')
    parser.add_argument('--save_images', action='store_true',
                       help='保存推理结果图片')
    parser.add_argument('--skip_gb', action='store_true',
                       help='跳过 GB 前置检测')
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = TamperDetectionTester(args.server, args.output)
    
    # 检查服务
    print("检查服务状态...")
    if not tester.check_service():
        print("✗ 服务不可用，请先启动服务")
        sys.exit(1)
    
    # 执行测试
    if args.mode == 'single':
        if not args.image:
            print("✗ 单张测试需要指定 --image 参数")
            sys.exit(1)
        tester.test_single(args.image, args.algorithm, args.save_images, args.mask)
    
    elif args.mode == 'batch':
        if args.algorithm == 'all':
            tester.compare_algorithms(args.data_dir, ['ela', 'dct', 'fusion', 'ml'], args.save_images)
        else:
            tester.test_batch(args.data_dir, args.algorithm, args.save_images)
    
    elif args.mode == 'compare':
        tester.compare_algorithms(args.data_dir, ['ela', 'dct', 'fusion', 'ml'], args.save_images)


if __name__ == '__main__':
    main()