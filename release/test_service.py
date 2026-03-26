#!/usr/bin/env python3
"""
图像篡改检测服务测试脚本 - 增强版

功能:
1. 单张图片测试
2. 批量图片测试
3. 多算法对比测试
4. 生成完整测试报告 (效果 + 性能 + 资源消耗)
5. 生成发版规格数据
6. 保存推理结果图片

数据集结构:
|-- test
|   |-- images
|   `-- masks

使用方法:
    # 单张测试
    python test_service.py --mode single --image test.jpg --algorithm ml
    
    # 批量测试
    python test_service.py --mode batch --data_dir ./test --algorithm ml
    
    # 全算法对比测试 (采集发版数据)
    python test_service.py --mode release --data_dir ./test --dataset_name Forgery-Internal-TestSet
    
    # 指定多个数据集测试
    python test_service.py --mode release --data_dirs ./test1,./test2 --dataset_names Dataset1,Dataset2
"""

import os
import sys
import json
import base64
import argparse
import time
import platform
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import requests
import cv2
import numpy as np

# 尝试导入资源监控库
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# 尝试导入GPU监控
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class SystemInfo:
    """系统信息采集器"""
    
    @staticmethod
    def get_os_info() -> Dict:
        """获取操作系统信息"""
        return {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
        }
    
    @staticmethod
    def get_cpu_info() -> Dict:
        """获取CPU信息"""
        info = {
            'cpu_count': os.cpu_count(),
            'cpu_percent': 0,
        }
        
        if HAS_PSUTIL:
            info['cpu_percent'] = psutil.cpu_percent(interval=0.1)
            try:
                freq = psutil.cpu_freq()
                if freq:
                    info['cpu_freq_current'] = freq.current
                    info['cpu_freq_max'] = freq.max
            except:
                pass
        
        return info
    
    @staticmethod
    def get_memory_info() -> Dict:
        """获取内存信息"""
        info = {
            'total_gb': 0,
            'available_gb': 0,
            'used_gb': 0,
            'percent': 0,
        }
        
        if HAS_PSUTIL:
            mem = psutil.virtual_memory()
            info['total_gb'] = round(mem.total / (1024**3), 2)
            info['available_gb'] = round(mem.available / (1024**3), 2)
            info['used_gb'] = round(mem.used / (1024**3), 2)
            info['percent'] = mem.percent
        
        return info
    
    @staticmethod
    def get_gpu_info() -> List[Dict]:
        """获取GPU信息"""
        gpus = []
        
        if HAS_TORCH and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu = {
                    'index': i,
                    'name': torch.cuda.get_device_name(i),
                    'total_memory_gb': round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2),
                }
                
                # 当前显存使用
                try:
                    allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    reserved = torch.cuda.memory_reserved(i) / (1024**3)
                    gpu['memory_allocated_gb'] = round(allocated, 2)
                    gpu['memory_reserved_gb'] = round(reserved, 2)
                except:
                    pass
                
                gpus.append(gpu)
        
        # 尝试使用nvidia-smi获取更多信息
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free', 
                                    '--format=csv,noheader,nounits'], 
                                   capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                for i, line in enumerate(result.stdout.strip().split('\n')):
                    parts = [x.strip() for x in line.split(',')]
                    if len(parts) >= 4:
                        if i < len(gpus):
                            gpus[i]['nvidia_name'] = parts[0]
                            gpus[i]['nvidia_total_mb'] = int(parts[1])
                            gpus[i]['nvidia_used_mb'] = int(parts[2])
                            gpus[i]['nvidia_free_mb'] = int(parts[3])
                        else:
                            gpus.append({
                                'index': i,
                                'name': parts[0],
                                'total_memory_mb': int(parts[1]),
                                'used_memory_mb': int(parts[2]),
                                'free_memory_mb': int(parts[3]),
                            })
        except:
            pass
        
        return gpus
    
    @staticmethod
    def get_python_info() -> Dict:
        """获取Python环境信息"""
        return {
            'python_version': platform.python_version(),
            'python_implementation': platform.python_implementation(),
        }
    
    @staticmethod
    def get_all_info() -> Dict:
        """获取所有系统信息"""
        return {
            'os': SystemInfo.get_os_info(),
            'cpu': SystemInfo.get_cpu_info(),
            'memory': SystemInfo.get_memory_info(),
            'gpu': SystemInfo.get_gpu_info(),
            'python': SystemInfo.get_python_info(),
            'timestamp': datetime.now().isoformat(),
        }


class ResourceMonitor:
    """资源监控器"""
    
    def __init__(self):
        self.start_time = None
        self.memory_samples = []
        self.cpu_samples = []
        self.gpu_memory_samples = []
        self._monitoring = False
    
    def start(self):
        """开始监控"""
        self.start_time = time.time()
        self.memory_samples = []
        self.cpu_samples = []
        self.gpu_memory_samples = []
        self._monitoring = True
    
    def sample(self):
        """采样一次"""
        if not self._monitoring:
            return
        
        if HAS_PSUTIL:
            self.memory_samples.append(psutil.virtual_memory().percent)
            self.cpu_samples.append(psutil.cpu_percent(interval=0.01))
        
        if HAS_TORCH and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                self.gpu_memory_samples.append({
                    'device': i,
                    'allocated_gb': torch.cuda.memory_allocated(i) / (1024**3),
                    'reserved_gb': torch.cuda.memory_reserved(i) / (1024**3),
                })
    
    def stop(self) -> Dict:
        """停止监控并返回统计"""
        self._monitoring = False
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        result = {
            'elapsed_seconds': round(elapsed, 2),
        }
        
        if self.memory_samples:
            result['memory_percent_avg'] = round(np.mean(self.memory_samples), 2)
            result['memory_percent_max'] = round(max(self.memory_samples), 2)
        
        if self.cpu_samples:
            result['cpu_percent_avg'] = round(np.mean(self.cpu_samples), 2)
            result['cpu_percent_max'] = round(max(self.cpu_samples), 2)
        
        if self.gpu_memory_samples:
            # 取最后一个设备的最大值
            max_allocated = max(s['allocated_gb'] for s in self.gpu_memory_samples)
            max_reserved = max(s['reserved_gb'] for s in self.gpu_memory_samples)
            result['gpu_memory_allocated_max_gb'] = round(max_allocated, 2)
            result['gpu_memory_reserved_max_gb'] = round(max_reserved, 2)
        
        return result


class TamperDetectionTester:
    """篡改检测服务测试器 - 增强版"""
    
    def __init__(self, server_url: str = "http://localhost:8000", output_dir: str = "./test_results"):
        """
        初始化测试器
        
        Args:
            server_url: 服务地址
            output_dir: 结果输出目录
        """
        self.server_url = server_url
        self.output_dir = output_dir
        self.timeout = 120
        self.resource_monitor = ResourceMonitor()
        
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
        """单张图片检测"""
        with open(image_path, 'rb') as f:
            image_base64 = base64.b64encode(f.read()).decode('utf-8')
        
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
                error_result = {
                    'status': f'HTTP_{response.status_code}',
                    'message': response.text[:500],
                    'request_time': request_time
                }
                print(f"\n[ERROR] HTTP {response.status_code}: {response.text[:200]}")
                return error_result
        except Exception as e:
            print(f"\n[ERROR] 请求异常: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'request_time': time.time() - start_time
            }
    
    def check_mask_label(self, mask_path: str) -> int:
        """根据 mask 判断标签"""
        if not os.path.exists(mask_path):
            return 1
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return 1
        
        if mask.sum() == 0:
            return 0
        else:
            return 1
    
    def decode_mask_base64(self, mask_base64: str) -> np.ndarray:
        """解码 Base64 掩码"""
        if not mask_base64:
            return None
        
        try:
            mask_bytes = base64.b64decode(mask_base64)
            mask = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
            return mask
        except:
            return None
    
    def compute_pixel_metrics(self, pred_mask: np.ndarray, true_mask: np.ndarray, 
                              threshold: int = 127) -> Dict:
        """计算像素级指标"""
        if pred_mask.shape != true_mask.shape:
            pred_mask = cv2.resize(pred_mask, (true_mask.shape[1], true_mask.shape[0]))
        
        pred_binary = (pred_mask > threshold).astype(np.uint8)
        true_binary = (true_mask > threshold).astype(np.uint8)
        
        tp = int(np.sum((pred_binary == 1) & (true_binary == 1)))
        tn = int(np.sum((pred_binary == 0) & (true_binary == 0)))
        fp = int(np.sum((pred_binary == 1) & (true_binary == 0)))
        fn = int(np.sum((pred_binary == 0) & (true_binary == 1)))
        
        intersection = tp
        union = tp + fp + fn
        iou = intersection / union if union > 0 else 0.0
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        pixel_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        pixel_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        pixel_f1 = 2 * pixel_precision * pixel_recall / (pixel_precision + pixel_recall) \
            if (pixel_precision + pixel_recall) > 0 else 0.0
        total_pixels = tp + tn + fp + fn
        pixel_accuracy = (tp + tn) / total_pixels if total_pixels > 0 else 0.0
        
        return {
            'iou': round(iou, 4),
            'dice': round(dice, 4),
            'pixel_precision': round(pixel_precision, 4),
            'pixel_recall': round(pixel_recall, 4),
            'pixel_f1': round(pixel_f1, 4),
            'pixel_accuracy': round(pixel_accuracy, 4),
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        }
    
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
        print("\n" + "=" * 60, flush=True)
        print("单张图片测试", flush=True)
        print("=" * 60, flush=True)
        
        if not os.path.exists(image_path):
            print(f"✗ 图片不存在: {image_path}")
            return
        
        print(f"图片: {image_path}", flush=True)
        print(f"算法: {algorithm}", flush=True)
        print(f"保存图片: {save_image}", flush=True)
        print("-" * 60, flush=True)
        
        # 判断真实标签
        true_label = None
        if mask_path and os.path.exists(mask_path):
            true_label = self.check_mask_label(mask_path)
            print(f"真实标签: {'篡改' if true_label else '正常'}")
        
        # 检测
        print("\n发送检测请求...", flush=True)
        result = self.detect_single(image_path, algorithm)
        
        # 打印结果
        print(f"\n检测结果:", flush=True)
        status = result.get('status', '')
        print(f"  状态码: {status}", flush=True)
        
        if status.startswith('0001'):
            is_tampered = result.get('is_tampered', False)
            confidence = result.get('confidence', 0)
            processing_time = result.get('processing_time', 0)
            
            print(f"  是否篡改: {'是' if is_tampered else '否'}", flush=True)
            print(f"  置信度: {confidence:.4f}", flush=True)
            print(f"  处理时间: {processing_time*1000:.2f}ms", flush=True)
            print(f"  请求时间: {result.get('request_time', 0)*1000:.2f}ms", flush=True)
            
            # 判断正确性
            if true_label is not None:
                pred_label = 1 if is_tampered else 0
                is_correct = pred_label == true_label
                print(f"  判断: {'✓ 正确' if is_correct else '✗ 错误'}", flush=True)
            
            # 篡改区域
            tamper_regions = result.get('tamper_regions')
            if tamper_regions:
                print(f"  篡改区域数: {len(tamper_regions)}", flush=True)
                for i, region in enumerate(tamper_regions[:3]):
                    print(f"    区域{i+1}: {region}", flush=True)
            
            # 保存结果图片
            if save_image:
                output_name = f"{os.path.splitext(os.path.basename(image_path))[0]}_{algorithm}_result.jpg"
                output_path = os.path.join(self.output_dir, 'images', output_name)
                
                print(f"\n保存结果图片...", flush=True)
                success = self._save_result_image(image_path, result, output_path, true_label, mask_path)
                if success:
                    print(f"  ✓ 结果图片已保存: {output_path}", flush=True)
                else:
                    print(f"  ✗ 保存失败", flush=True)
            
            return {
                'success': True,
                'is_tampered': is_tampered,
                'confidence': confidence,
                'processing_time': processing_time,
                'status': status
            }
        else:
            # 错误情况
            error_msg = status if status else result.get('message', 'Unknown error')
            print(f"  ✗ 检测失败: {error_msg}", flush=True)
            return {
                'success': False,
                'error': error_msg,
                'status': status
            }
    
    def _save_result_image(self, image_path: str, result: Dict, output_path: str,
                           true_label: int = None, mask_path: str = None):
        """保存检测结果图片"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"  读取原图失败: {image_path}")
                return False
            
            h, w = image.shape[:2]
            
            # 方式1: 使用服务返回的 marked_image_base64 (优先)
            marked_image_base64 = result.get('marked_image_base64')
            if marked_image_base64:
                print(f"  使用服务返回的标注图片", flush=True)
                marked_image = self.decode_mask_base64(marked_image_base64)
                if marked_image is not None:
                    # 转换为彩色图（如果是灰度图）
                    if len(marked_image.shape) == 2:
                        marked_image = cv2.cvtColor(marked_image, cv2.COLOR_GRAY2BGR)
                    
                    # 创建画布 (原图 + 标注图)
                    canvas = np.ones((h, w * 2 + 20, 3), dtype=np.uint8) * 255
                    canvas[:h, :w] = image
                    canvas[:, w:w+20] = 200
                    canvas[:h, w+20:] = marked_image
                    
                    # 添加文字标注
                    self._add_text_annotations(canvas[:, w+20:], result, true_label, h, w)
                    
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    cv2.imwrite(output_path, canvas)
                    return True
            
            # 方式2: 使用服务返回的 mask_base64 创建掩码覆盖
            mask_base64 = result.get('mask_base64')
            if mask_base64:
                print(f"  使用服务返回的掩码图片", flush=True)
                mask = self.decode_mask_base64(mask_base64)
                if mask is not None:
                    # 创建掩码覆盖图
                    overlay = image.copy()
                    
                    # 红色标注篡改区域
                    overlay[mask > 127] = [0, 0, 255]
                    
                    # 半透明混合
                    result_image = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
                    
                    # 创建画布
                    canvas = np.ones((h, w * 2 + 20, 3), dtype=np.uint8) * 255
                    canvas[:h, :w] = image
                    canvas[:, w:w+20] = 200
                    canvas[:h, w+20:] = result_image
                    
                    # 添加文字标注
                    self._add_text_annotations(canvas[:, w+20:], result, true_label, h, w)
                    
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    cv2.imwrite(output_path, canvas)
                    return True
            
            # 方式3: 使用 tamper_regions 绘制矩形
            tamper_regions = result.get('tamper_regions')
            if tamper_regions and len(tamper_regions) > 0:
                print(f"  使用篡改区域坐标绘制 (共{len(tamper_regions)}个区域)", flush=True)
                pred_image = image.copy()
                
                for region in tamper_regions:
                    x1, y1 = region['left_top']
                    x2, y2 = region['right_bottom']
                    cv2.rectangle(pred_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # 添加文字标注
                self._add_text_annotations(pred_image, result, true_label, h, w)
                
                # 创建画布
                canvas = np.ones((h, w * 2 + 20, 3), dtype=np.uint8) * 255
                canvas[:h, :w] = image
                canvas[:, w:w+20] = 200
                canvas[:h, w+20:] = pred_image
                
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, canvas)
                return True
            
            # 方式4: 没有篡改区域，只显示原图 + 预测结果文字
            print(f"  无篡改区域标注，显示原图 + 预测结果", flush=True)
            pred_image = image.copy()
            self._add_text_annotations(pred_image, result, true_label, h, w)
            
            canvas = np.ones((h, w * 2 + 20, 3), dtype=np.uint8) * 255
            canvas[:h, :w] = image
            canvas[:, w:w+20] = 200
            canvas[:h, w+20:] = pred_image
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, canvas)
            return True
            
        except Exception as e:
            print(f"  保存图片失败: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return False
    
    def _add_text_annotations(self, image: np.ndarray, result: Dict, 
                              true_label: int, h: int, w: int):
        """添加文字标注"""
        is_tampered = result.get('is_tampered', False)
        confidence = result.get('confidence', 0)
        
        pred_text = "TAMPERED" if is_tampered else "GOOD"
        color = (0, 0, 255) if is_tampered else (0, 255, 0)
        
        cv2.putText(image, f"Pred: {pred_text}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(image, f"Conf: {confidence:.3f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        if true_label is not None:
            true_text = "TAMPERED" if true_label else "GOOD"
            true_color = (0, 100, 255) if true_label else (100, 255, 0)
            cv2.putText(image, f"True: {true_text}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, true_color, 2)
            
            pred_label = 1 if is_tampered else 0
            is_correct = pred_label == true_label
            correct_text = "CORRECT" if is_correct else "WRONG"
            correct_color = (0, 255, 0) if is_correct else (0, 0, 255)
            cv2.putText(image, correct_text, (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, correct_color, 2)
    
    def test_batch(self, data_dir: str, algorithm: str = "ml", 
                   save_images: bool = False, dataset_name: str = None) -> Dict:
        """
        批量图片测试
        
        Args:
            data_dir: 数据目录
            algorithm: 算法名称
            save_images: 是否保存结果图片
            dataset_name: 数据集名称
        """
        data_dir = os.path.abspath(data_dir)
        images_dir = os.path.join(data_dir, 'images')
        masks_dir = os.path.join(data_dir, 'masks')
        
        if not os.path.exists(images_dir):
            print(f"✗ 图片目录不存在: {images_dir}")
            return None
        
        # 获取图片列表
        image_files = sorted([f for f in os.listdir(images_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
        total = len(image_files)
        
        if total == 0:
            print(f"✗ 未找到图片")
            return None
        
        print(f"\n{'='*60}")
        print(f"批量测试: {dataset_name or data_dir}")
        print(f"{'='*60}")
        print(f"算法: {algorithm}")
        print(f"图片总数: {total}")
        print(f"保存图片: {'是' if save_images else '否'}")
        print("-" * 60)
        
        # 开始资源监控
        self.resource_monitor.start()
        
        # 统计变量
        results = []
        processing_times = []
        tampered_count = 0
        good_count = 0
        tp = tn = fp = fn = 0
        
        pixel_metrics_list = []
        total_pixel_tp = total_pixel_tn = total_pixel_fp = total_pixel_fn = 0
        
        tampered_confidences = []
        normal_confidences = []
        
        start_time = time.time()
        
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(images_dir, image_file)
            
            # 查找对应的 mask
            mask_name = os.path.splitext(image_file)[0] + '.png'
            mask_path = os.path.join(masks_dir, mask_name)
            if not os.path.exists(mask_path):
                mask_name = os.path.splitext(image_file)[0] + '.jpg'
                mask_path = os.path.join(masks_dir, mask_name)
            
            true_label = self.check_mask_label(mask_path)
            if true_label == 1:
                tampered_count += 1
            else:
                good_count += 1
            
            # 检测
            result = self.detect_single(image_path, algorithm)
            
            # 资源采样
            self.resource_monitor.sample()
            
            # 检查返回结果
            status = result.get('status', '')
            
            # 成功状态码以 '0001' 开头
            if status.startswith('0001'):
                processing_time = result.get('processing_time', 0)
                processing_times.append(processing_time)
                
                is_tampered = result.get('is_tampered', False)
                confidence = result.get('confidence', 0)
                pred_label = 1 if is_tampered else 0
                
                # 更新混淆矩阵
                if true_label == 1 and pred_label == 1:
                    tp += 1
                elif true_label == 0 and pred_label == 0:
                    tn += 1
                elif true_label == 0 and pred_label == 1:
                    fp += 1
                else:
                    fn += 1
                
                # 置信度统计
                if true_label == 1:
                    tampered_confidences.append(confidence)
                else:
                    normal_confidences.append(confidence)
                
                # 像素级指标 (如果有mask和预测mask)
                if os.path.exists(mask_path) and pred_label == 1:
                    true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    pred_mask_base64 = result.get('mask_base64')
                    
                    if true_mask is not None and pred_mask_base64:
                        pred_mask = self.decode_mask_base64(pred_mask_base64)
                        if pred_mask is not None:
                            pixel_metrics = self.compute_pixel_metrics(pred_mask, true_mask)
                            pixel_metrics_list.append(pixel_metrics)
                            total_pixel_tp += pixel_metrics['tp']
                            total_pixel_tn += pixel_metrics['tn']
                            total_pixel_fp += pixel_metrics['fp']
                            total_pixel_fn += pixel_metrics['fn']
                
                results.append({
                    'image': image_file,
                    'true_label': true_label,
                    'pred_label': pred_label,
                    'confidence': confidence,
                    'processing_time': processing_time,
                })
            else:
                # 错误状态码: 0000/0002/0004/0007
                # 服务返回的错误信息在 status 字段中，格式: "XXXX:错误描述"
                error_msg = status if status else result.get('message', 'Unknown error')
                
                # 打印前几个错误的详细信息
                error_count = sum(1 for r in results if r.get('pred_label') == -1)
                if error_count < 5:
                    print(f"\n  ⚠️ 图片 {image_file} 检测失败: {error_msg}")
                
                fn += 1 if true_label == 1 else 0
                fp += 1 if true_label == 0 else 0
                results.append({
                    'image': image_file,
                    'true_label': true_label,
                    'pred_label': -1,
                    'error': error_msg,
                })
            
            # 进度显示
            if (i + 1) % 100 == 0 or (i + 1) == total:
                elapsed = time.time() - start_time
                fps = (i + 1) / elapsed
                eta = (total - i - 1) / fps if fps > 0 else 0
                print(f"\r  [{i+1}/{total}] {fps:.1f} FPS | 耗时: {elapsed:.1f}s | 预计: {eta:.1f}s", end='')
        
        print()
        
        total_time = time.time() - start_time
        
        # 停止资源监控
        resource_stats = self.resource_monitor.stop()
        
        # 计算指标
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        avg_time = np.mean(processing_times) if processing_times else 0
        
        avg_tampered_conf = np.mean(tampered_confidences) if tampered_confidences else 0
        avg_normal_conf = np.mean(normal_confidences) if normal_confidences else 0
        
        # 像素级指标汇总
        pixel_summary = None
        if pixel_metrics_list:
            avg_iou = np.mean([m['iou'] for m in pixel_metrics_list])
            avg_dice = np.mean([m['dice'] for m in pixel_metrics_list])
            avg_pixel_f1 = np.mean([m['pixel_f1'] for m in pixel_metrics_list])
            avg_pixel_prec = np.mean([m['pixel_precision'] for m in pixel_metrics_list])
            avg_pixel_rec = np.mean([m['pixel_recall'] for m in pixel_metrics_list])
            avg_pixel_acc = np.mean([m['pixel_accuracy'] for m in pixel_metrics_list])
            
            # 整体像素级指标
            overall_iou = total_pixel_tp / (total_pixel_tp + total_pixel_fp + total_pixel_fn) \
                if (total_pixel_tp + total_pixel_fp + total_pixel_fn) > 0 else 0
            overall_dice = 2 * total_pixel_tp / (2 * total_pixel_tp + total_pixel_fp + total_pixel_fn) \
                if (2 * total_pixel_tp + total_pixel_fp + total_pixel_fn) > 0 else 0
            
            pixel_summary = {
                'evaluated_images': len(pixel_metrics_list),
                'avg_iou': round(avg_iou, 4),
                'avg_dice': round(avg_dice, 4),
                'avg_pixel_f1': round(avg_pixel_f1, 4),
                'avg_pixel_precision': round(avg_pixel_prec, 4),
                'avg_pixel_recall': round(avg_pixel_rec, 4),
                'avg_pixel_accuracy': round(avg_pixel_acc, 4),
                'overall_iou': round(overall_iou, 4),
                'overall_dice': round(overall_dice, 4),
                'total_tp': total_pixel_tp,
                'total_tn': total_pixel_tn,
                'total_fp': total_pixel_fp,
                'total_fn': total_pixel_fn,
            }
        
        # 构建报告
        report = {
            'test_info': {
                'data_dir': data_dir,
                'dataset_name': dataset_name or os.path.basename(data_dir),
                'algorithm': algorithm,
                'test_time': datetime.now().isoformat(),
                'total_images': total,
                'tested_images': len(results),
                'tampered_images': tampered_count,
                'good_images': good_count,
            },
            'performance': {
                'total_time': round(total_time, 2),
                'avg_time': round(avg_time, 4),
                'avg_processing_time_ms': round(avg_time * 1000, 2),
                'min_time_ms': round(min(processing_times) * 1000, 2) if processing_times else 0,
                'max_time_ms': round(max(processing_times) * 1000, 2) if processing_times else 0,
                'p50_time_ms': round(np.percentile(processing_times, 50) * 1000, 2) if processing_times else 0,
                'p95_time_ms': round(np.percentile(processing_times, 95) * 1000, 2) if processing_times else 0,
                'fps': round(total / total_time, 2) if total_time > 0 else 0,
            },
            'classification_metrics': {
                'accuracy': round(accuracy, 4),
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1': round(f1, 4),
                'fpr': round(fpr, 4),
                'fnr': round(fnr, 4),
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            },
            'pixel_metrics': pixel_summary,
            'confidence_stats': {
                'avg_tampered_confidence': round(avg_tampered_conf, 4),
                'avg_normal_confidence': round(avg_normal_conf, 4),
                'tampered_count': len(tampered_confidences),
                'normal_count': len(normal_confidences),
            },
            'resource_usage': resource_stats,
            'detailed_results': results[:100],  # 只保存前100条详细结果
        }
        
        self._print_report(report)
        
        # 保存报告
        report_path = os.path.join(self.output_dir, 
            f"test_report_{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n✓ 测试报告已保存: {report_path}")
        
        return report
    
    def _print_report(self, report: Dict):
        """打印测试报告"""
        print("\n" + "=" * 60)
        print("测试报告")
        print("=" * 60)
        
        info = report['test_info']
        print(f"\n【测试信息】")
        print(f"  数据集: {info.get('dataset_name', info['data_dir'])}")
        print(f"  算法: {info['algorithm']}")
        print(f"  时间: {info['test_time']}")
        print(f"  图片: {info['tested_images']}/{info['total_images']} (篡改:{info['tampered_images']}, 正常:{info['good_images']})")
        
        cls = report['classification_metrics']
        print(f"\n【分类指标】")
        print(f"  Accuracy:  {cls['accuracy']:.4f}")
        print(f"  Precision: {cls['precision']:.4f}")
        print(f"  Recall:    {cls['recall']:.4f}")
        print(f"  F1-score:  {cls['f1']:.4f}")
        print(f"  FPR: {cls['fpr']:.4f} | FNR: {cls['fnr']:.4f}")
        print(f"  混淆矩阵: TP={cls['tp']} TN={cls['tn']} FP={cls['fp']} FN={cls['fn']}")
        
        pixel = report.get('pixel_metrics')
        if pixel:
            print(f"\n【像素级指标】(评估{pixel['evaluated_images']}张)")
            print(f"  IoU:  {pixel['avg_iou']:.4f} (Overall: {pixel['overall_iou']:.4f})")
            print(f"  Dice: {pixel['avg_dice']:.4f} (Overall: {pixel['overall_dice']:.4f})")
            print(f"  Pixel-F1: {pixel['avg_pixel_f1']:.4f}")
            print(f"  Pixel-Precision: {pixel['avg_pixel_precision']:.4f}")
            print(f"  Pixel-Recall: {pixel['avg_pixel_recall']:.4f}")
        
        perf = report['performance']
        print(f"\n【性能指标】")
        print(f"  总耗时: {perf['total_time']:.2f}s")
        print(f"  平均处理: {perf['avg_processing_time_ms']:.2f}ms")
        print(f"  P50/P95: {perf['p50_time_ms']:.2f}ms / {perf['p95_time_ms']:.2f}ms")
        print(f"  FPS: {perf['fps']:.2f}")
        
        res = report.get('resource_usage', {})
        if res:
            print(f"\n【资源消耗】")
            print(f"  耗时: {res.get('elapsed_seconds', 0)}s")
            if 'memory_percent_max' in res:
                print(f"  内存峰值: {res['memory_percent_max']:.1f}%")
            if 'cpu_percent_max' in res:
                print(f"  CPU峰值: {res['cpu_percent_max']:.1f}%")
            if 'gpu_memory_allocated_max_gb' in res:
                print(f"  GPU显存峰值: {res['gpu_memory_allocated_max_gb']:.2f}GB")
    
    def compare_algorithms(self, data_dir: str, algorithms: List[str] = None, 
                          save_images: bool = False, dataset_name: str = None) -> Dict:
        """多算法对比测试"""
        if algorithms is None:
            algorithms = ['ela', 'dct', 'fusion', 'ml']
        
        print("\n" + "=" * 60)
        print("多算法对比测试")
        print("=" * 60)
        
        reports = {}
        for algo in algorithms:
            print(f"\n>>> 测试算法: {algo}")
            report = self.test_batch(data_dir, algo, save_images, dataset_name)
            reports[algo] = report
        
        # 打印对比表
        print("\n" + "=" * 60)
        print("算法对比结果")
        print("=" * 60)
        
        print(f"\n【分类指标对比】")
        print(f"{'算法':<10} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'时间(ms)':<10}")
        print("-" * 60)
        
        for algo, report in reports.items():
            if report:
                cls = report['classification_metrics']
                perf = report['performance']
                print(f"{algo:<10} {cls['accuracy']:<8.4f} {cls['precision']:<8.4f} "
                      f"{cls['recall']:<8.4f} {cls['f1']:<8.4f} {perf['avg_processing_time_ms']:<10.2f}")
        
        # 像素级对比
        has_pixel = any(r.get('pixel_metrics') for r in reports.values() if r)
        if has_pixel:
            print(f"\n【像素级指标对比】")
            print(f"{'算法':<10} {'IoU':<8} {'Dice':<8} {'P-F1':<8} {'P-Prec':<8} {'P-Rec':<8}")
            print("-" * 60)
            
            for algo, report in reports.items():
                if report and report.get('pixel_metrics'):
                    pixel = report['pixel_metrics']
                    print(f"{algo:<10} {pixel['avg_iou']:<8.4f} {pixel['avg_dice']:<8.4f} "
                          f"{pixel['avg_pixel_f1']:<8.4f} {pixel['avg_pixel_precision']:<8.4f} "
                          f"{pixel['avg_pixel_recall']:<8.4f}")
        
        # 保存对比报告
        compare_path = os.path.join(self.output_dir, 
            f"compare_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(compare_path, 'w', encoding='utf-8') as f:
            json.dump(reports, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n✓ 对比报告已保存: {compare_path}")
        
        return reports
    
    def generate_release_report(self, data_dirs: List[str], dataset_names: List[str] = None,
                                algorithms: List[str] = None) -> Dict:
        """
        生成发版规格数据报告
        
        Args:
            data_dirs: 数据集目录列表
            dataset_names: 数据集名称列表
            algorithms: 算法列表
        """
        if algorithms is None:
            algorithms = ['ela', 'dct', 'fusion', 'ml']
        
        if dataset_names is None:
            dataset_names = [os.path.basename(d) for d in data_dirs]
        
        print("\n" + "=" * 60)
        print("生成发版规格数据报告")
        print("=" * 60)
        
        # 收集系统信息
        print("\n>>> 收集系统信息...")
        system_info = SystemInfo.get_all_info()
        
        # 收集各数据集各算法的测试结果
        all_results = {}
        
        for data_dir, dataset_name in zip(data_dirs, dataset_names):
            print(f"\n>>> 测试数据集: {dataset_name}")
            all_results[dataset_name] = {}
            
            for algo in algorithms:
                print(f"\n  >>> 算法: {algo}")
                report = self.test_batch(data_dir, algo, False, dataset_name)
                all_results[dataset_name][algo] = report
        
        # 生成汇总报告
        release_report = {
            'report_info': {
                'generated_at': datetime.now().isoformat(),
                'report_type': 'release_specification',
            },
            'system_info': system_info,
            'datasets': {},
            'algorithm_summary': {},
        }
        
        # 按数据集组织
        for dataset_name, algo_results in all_results.items():
            release_report['datasets'][dataset_name] = {
                algo: {
                    'classification': r['classification_metrics'] if r else None,
                    'pixel': r.get('pixel_metrics') if r else None,
                    'performance': r['performance'] if r else None,
                    'resource': r.get('resource_usage') if r else None,
                    'test_info': r['test_info'] if r else None,
                }
                for algo, r in algo_results.items()
            }
        
        # 算法汇总 (跨数据集平均)
        for algo in algorithms:
            algo_reports = [all_results[ds][algo] for ds in dataset_names if all_results[ds].get(algo)]
            valid_reports = [r for r in algo_reports if r]
            
            if valid_reports:
                avg_acc = np.mean([r['classification_metrics']['accuracy'] for r in valid_reports])
                avg_prec = np.mean([r['classification_metrics']['precision'] for r in valid_reports])
                avg_rec = np.mean([r['classification_metrics']['recall'] for r in valid_reports])
                avg_f1 = np.mean([r['classification_metrics']['f1'] for r in valid_reports])
                avg_time = np.mean([r['performance']['avg_processing_time_ms'] for r in valid_reports])
                
                # 像素级平均
                pixel_reports = [r.get('pixel_metrics') for r in valid_reports if r.get('pixel_metrics')]
                avg_iou = np.mean([p['avg_iou'] for p in pixel_reports]) if pixel_reports else None
                avg_dice = np.mean([p['avg_dice'] for p in pixel_reports]) if pixel_reports else None
                
                release_report['algorithm_summary'][algo] = {
                    'avg_accuracy': round(avg_acc, 4),
                    'avg_precision': round(avg_prec, 4),
                    'avg_recall': round(avg_rec, 4),
                    'avg_f1': round(avg_f1, 4),
                    'avg_time_ms': round(avg_time, 2),
                    'avg_iou': round(avg_iou, 4) if avg_iou else None,
                    'avg_dice': round(avg_dice, 4) if avg_dice else None,
                }
        
        # 保存报告
        report_path = os.path.join(self.output_dir, 
            f"release_spec_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(release_report, f, indent=2, ensure_ascii=False, default=str)
        
        # 打印汇总
        print("\n" + "=" * 60)
        print("发版规格数据汇总")
        print("=" * 60)
        
        print(f"\n【系统环境】")
        print(f"  OS: {system_info['os']['system']} {system_info['os']['release']}")
        print(f"  CPU: {system_info['cpu']['cpu_count']} cores")
        print(f"  内存: {system_info['memory']['total_gb']}GB")
        if system_info['gpu']:
            for gpu in system_info['gpu']:
                print(f"  GPU: {gpu.get('name', 'Unknown')}")
        
        print(f"\n【算法性能汇总】")
        print(f"{'算法':<8} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'时间(ms)':<10} {'IoU':<8}")
        print("-" * 70)
        
        for algo, summary in release_report['algorithm_summary'].items():
            iou_str = f"{summary['avg_iou']:.4f}" if summary.get('avg_iou') else "N/A"
            print(f"{algo:<8} {summary['avg_accuracy']:<8.4f} {summary['avg_precision']:<8.4f} "
                  f"{summary['avg_recall']:<8.4f} {summary['avg_f1']:<8.4f} "
                  f"{summary['avg_time_ms']:<10.2f} {iou_str:<8}")
        
        print(f"\n✓ 发版报告已保存: {report_path}")
        
        return release_report


def main():
    parser = argparse.ArgumentParser(description='图像篡改检测服务测试脚本 - 增强版')
    parser.add_argument('--mode', type=str, 
                        choices=['single', 'batch', 'compare', 'release'], 
                        default='single',
                        help='测试模式: single/batch/compare/release')
    parser.add_argument('--image', type=str, help='单张测试图片路径')
    parser.add_argument('--data_dir', type=str, default='./test',
                        help='数据目录')
    parser.add_argument('--data_dirs', type=str, 
                        help='多个数据目录 (逗号分隔)')
    parser.add_argument('--dataset_name', type=str, help='数据集名称')
    parser.add_argument('--dataset_names', type=str, 
                        help='多个数据集名称 (逗号分隔)')
    parser.add_argument('--algorithm', type=str, default='ml',
                        help='算法: ela/dct/fusion/ml/all')
    parser.add_argument('--server', type=str, default='http://localhost:8000',
                        help='服务地址')
    parser.add_argument('--output', type=str, default='./test_results',
                        help='输出目录')
    parser.add_argument('--save_images', action='store_true',
                        help='保存结果图片')
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = TamperDetectionTester(args.server, args.output)
    
    # 检查服务
    print("检查服务状态...")
    if not tester.check_service():
        print("✗ 服务不可用，请先启动服务")
        sys.exit(1)
    
    if args.mode == 'single':
        if not args.image:
            print("✗ 请指定 --image 参数")
            sys.exit(1)
        tester.test_single(args.image, args.algorithm, args.save_images)
    
    elif args.mode == 'batch':
        algorithms = ['ela', 'dct', 'fusion', 'ml'] if args.algorithm == 'all' else [args.algorithm]
        if len(algorithms) == 1:
            tester.test_batch(args.data_dir, algorithms[0], args.save_images, args.dataset_name)
        else:
            tester.compare_algorithms(args.data_dir, algorithms, args.save_images, args.dataset_name)
    
    elif args.mode == 'compare':
        tester.compare_algorithms(args.data_dir, None, args.save_images, args.dataset_name)
    
    elif args.mode == 'release':
        data_dirs = args.data_dirs.split(',') if args.data_dirs else [args.data_dir]
        dataset_names = args.dataset_names.split(',') if args.dataset_names else None
        tester.generate_release_report(data_dirs, dataset_names)


if __name__ == '__main__':
    main()