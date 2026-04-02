#!/usr/bin/env python3
"""
图像篡改检测服务

用法:
    python server_forgrey.py
    
    # 或指定端口
    python server_forgrey.py --port 8000

功能:
    1. GB 分类器前置检测
    2. 多算法支持 (ela/dct/fusion/ml)
    3. 像素级篡改区域定位
    4. 篡改区域坐标输出
    5. 在原图上绘制篡改区域矩形框
    6. 自动日志管理 (保存到 ./logs，每30天清理)
    7. 默认8并发支持

流程:
    1. GB 分类器判断图片是否篡改 (前置检测)
    2. 如果篡改，使用指定算法定位篡改区域
    3. 在原图上绘制矩形框标注篡改区域

入参:
    - image_base64: 图片 Base64 编码
    - algorithm: 算法名称 (ela/dct/fusion/ml)
    - skip_gb: 跳过 GB 前置检测 (可选)

出参:
    - status: 状态码
    - is_tampered: 是否篡改
    - confidence: 置信度
    - gb_confidence: GB 分类器置信度
    - mask_base64: 篡改区域掩码 Base64 (二值图)
    - marked_image_base64: 原图标记篡改区域图片 Base64
    - tamper_regions: 篡改区域矩形坐标列表
"""

from __future__ import annotations

import os
import sys
import base64
import pickle
import tempfile
import time
import argparse
import logging
import glob
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from logging.handlers import RotatingFileHandler
from pathlib import Path

import cv2
import numpy as np

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
import uvicorn

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 导入检测器
from algorithms.ela_detector import ELADetector
from algorithms.dct_detector import DCTBlockDetector
from algorithms.fusion_detector import AdaptiveFusion
from algorithms.ml_detector import MLDetectorGPU
from algorithms.features import extract_all_features

# 状态码定义
STATUS_CODES = {
    "0000": "未知异常.",
    "0001": "解析成功.",
    "0002": "base64解码异常.",
    "0004": "参数格式错误.",
    "0007": "请求参数内容错误或为空."
}

# 日志配置
logger = logging.getLogger("forgery_server")


def setup_logging():
    """
    配置日志系统
    
    日志保存到 ./logs 目录，自动管理：
    - 日志轮转: 10MB 一个文件，保留5个
    - 定期清理: 每30天删除一次旧日志
    """
    # 日志目录
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 日志文件
    log_file = os.path.join(log_dir, 'server.log')
    
    # 日志格式
    log_format = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # 创建 formatter
    formatter = logging.Formatter(log_format, date_format)
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # 文件处理器 (日志轮转)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # 配置根日志
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    logger.info(f"日志文件: {log_file}")
    
    # 定期清理旧日志 (每30天)
    clean_old_logs(log_dir, days=30)
    
    return log_file


def clean_old_logs(log_dir: str, days: int = 30):
    """
    清理旧日志文件
    
    Args:
        log_dir: 日志目录
        days: 保留天数
    """
    try:
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # 查找所有日志文件
        log_files = glob.glob(os.path.join(log_dir, '*.log*'))
        
        cleaned = 0
        for log_file in log_files:
            try:
                # 获取文件修改时间
                mtime = datetime.fromtimestamp(os.path.getmtime(log_file))
                if mtime < cutoff_date:
                    os.remove(log_file)
                    cleaned += 1
            except Exception as e:
                logger.warning(f"清理日志文件失败 {log_file}: {e}")
        
        if cleaned > 0:
            logger.info(f"清理了 {cleaned} 个旧日志文件 (超过 {days} 天)")
    except Exception as e:
        logger.warning(f"日志清理失败: {e}")


# 创建 FastAPI 应用
app = FastAPI(
    title="图像篡改检测服务",
    description="检测图像是否经过拼接、复制粘贴、修饰等篡改操作\n\n流程: GB分类器(前置) → 区域定位算法",
    version="2.2.0"
)

# 初始化检测器
detectors = {}
gb_model = None
gb_scaler = None
gb_threshold = 0.85  # 提高阈值以降低误报率 (训练数据篡改:正常=14.5:1)


def load_gb_classifier():
    """加载 GB 分类器"""
    global gb_model, gb_scaler, gb_threshold
    
    if gb_model is not None:
        return True
    
    model_path = os.path.join(PROJECT_ROOT, 'release', 'models', 'gb_classifier', 'model.pkl')
    scaler_path = os.path.join(PROJECT_ROOT, 'release', 'models', 'gb_classifier', 'scaler.pkl')
    config_path = os.path.join(PROJECT_ROOT, 'release', 'models', 'gb_classifier', 'config.json')
    
    if not os.path.exists(model_path):
        logger.warning(f"GB 分类器模型不存在: {model_path}")
        return False
    
    try:
        with open(model_path, 'rb') as f:
            gb_model = pickle.load(f)
        logger.info(f"GB 分类器已加载: {model_path}")
        
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                gb_scaler = pickle.load(f)
            logger.info(f"GB 标准化器已加载: {scaler_path}")
        
        # 加载配置
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
                threshold_from_config = config.get('optimal_threshold', 0.85)
                # 训练数据严重不平衡(篡改:正常=14.5:1)，强制使用高阈值
                gb_threshold = max(threshold_from_config, 0.85)
                logger.info(f"GB 最优阈值: {gb_threshold} (config={threshold_from_config})")
        
        return True
    except Exception as e:
        logger.error(f"GB 分类器加载失败: {e}")
        return False


def gb_classify(image_path: str) -> Dict:
    """
    GB 分类器前置检测
    
    Args:
        image_path: 图片路径
        
    Returns:
        {
            'is_tampered': bool,    # 是否篡改
            'confidence': float,    # 置信度
            'skipped': bool         # 是否跳过(模型未加载)
        }
    """
    global gb_model, gb_scaler, gb_threshold
    
    result = {
        'is_tampered': True,  # 默认认为篡改，继续后续检测
        'confidence': 1.0,
        'skipped': True
    }
    
    if gb_model is None:
        if not load_gb_classifier():
            return result
    
    result['skipped'] = False
    
    try:
        features = extract_all_features(image_path)
        if features is not None and gb_scaler is not None:
            features_scaled = gb_scaler.transform([features])
            proba = gb_model.predict_proba(features_scaled)[0, 1]
            result['is_tampered'] = bool(proba > gb_threshold)
            result['confidence'] = float(proba)
            logger.debug(f"GB 分类结果: is_tampered={result['is_tampered']}, confidence={result['confidence']:.4f}")
    except Exception as e:
        logger.error(f"GB 分类错误: {e}")
    
    return result


def get_ela_detector():
    """获取 ELA 检测器"""
    if 'ela' not in detectors:
        detectors['ela'] = ELADetector()
        logger.debug("ELA 检测器已初始化")
    return detectors['ela']


def get_dct_detector():
    """获取 DCT 检测器"""
    if 'dct' not in detectors:
        detectors['dct'] = DCTBlockDetector()
        logger.debug("DCT 检测器已初始化")
    return detectors['dct']


def get_fusion_detector():
    """获取融合检测器"""
    if 'fusion' not in detectors:
        detectors['fusion'] = AdaptiveFusion()
        logger.debug("融合检测器已初始化")
    return detectors['fusion']


def get_ml_detector():
    """获取 ML 检测器"""
    if 'ml' not in detectors:
        detectors['ml'] = MLDetectorGPU()
        logger.debug("ML 检测器已初始化")
    return detectors['ml']


# 响应模型
class TamperRegion(BaseModel):
    """篡改区域矩形"""
    left_top: List[int]      # 左上角坐标 [x, y]
    right_bottom: List[int]  # 右下角坐标 [x, y]


class DetectionResponse(BaseModel):
    """检测响应"""
    status: str                              # 状态码
    is_tampered: bool                        # 是否篡改
    confidence: float                        # 置信度 (0-1)
    gb_confidence: Optional[float]           # GB 分类器置信度
    mask_base64: Optional[str]               # 篡改区域掩码 Base64
    marked_image_base64: Optional[str]       # 原图标记篡改区域图片 Base64
    tamper_regions: Optional[List[TamperRegion]]  # 篡改区域矩形坐标列表
    algorithm: str                           # 使用的算法
    processing_time: float                   # 处理时间 (秒)


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    timestamp: str
    algorithms: list
    gb_classifier_loaded: bool


class ErrorResponse(BaseModel):
    """错误响应"""
    status: str
    message: str


# 全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理 - 返回0000状态码"""
    logger.error(f"全局异常: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": "0000:未知异常.",
            "message": str(exc)
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """参数验证异常 - 返回0004状态码"""
    logger.warning(f"参数验证错误: {exc}")
    return JSONResponse(
        status_code=400,
        content={
            "status": "0004:参数格式错误.",
            "message": str(exc)
        }
    )


# API 端点
@app.get("/", response_model=HealthResponse)
async def root():
    """根路径 - 健康检查"""
    return {
        "status": "0001:服务运行正常.",
        "timestamp": datetime.now().isoformat(),
        "algorithms": ["ela", "dct", "fusion", "ml"],
        "gb_classifier_loaded": gb_model is not None
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """健康检查"""
    return {
        "status": "0001:服务运行正常.",
        "timestamp": datetime.now().isoformat(),
        "algorithms": ["ela", "dct", "fusion", "ml"],
        "gb_classifier_loaded": gb_model is not None
    }


@app.post("/tamper_detection/v1/tamper_detect_img")
async def tamper_detect_img(
    image_base64: str = Form(default="", description="图片 Base64 编码"),
    algorithm: str = Form(default="ml", description="算法名称: ela/dct/fusion/ml"),
    skip_gb: bool = Form(default=False, description="跳过GB前置检测")
):
    """
    检测图片是否篡改
    
    流程:
    1. GB 分类器判断是否篡改 (前置检测)
    2. 如果篡改，使用指定算法定位区域
    3. 如果正常，直接返回不进行区域定位
    
    - **image_base64**: 图片 Base64 编码，支持带或不带 data URI 前缀
    - **algorithm**: 算法名称
        - ela: ELA 单特征检测
        - dct: DCT 单特征检测
        - fusion: 多特征融合检测
        - ml: 机器学习串联检测 (推荐)
    - **skip_gb**: 跳过 GB 前置检测，直接进行区域定位
    """
    start_time = time.time()
    request_id = f"{int(start_time * 1000)}"
    
    logger.info(f"[{request_id}] 收到检测请求: algorithm={algorithm}, skip_gb={skip_gb}")
    
    # 检查参数是否为空
    if not image_base64 or image_base64.strip() == "":
        logger.warning(f"[{request_id}] 参数为空")
        return JSONResponse(
            status_code=400,
            content={
                "status": "0007:请求参数内容错误或为空.",
                "message": "image_base64参数不能为空"
            }
        )
    
    # 检查算法参数
    if algorithm not in ['ela', 'dct', 'fusion', 'ml']:
        logger.warning(f"[{request_id}] 未知算法: {algorithm}")
        return JSONResponse(
            status_code=400,
            content={
                "status": "0004:参数格式错误.",
                "message": f"未知的算法: {algorithm}，可选值: ela/dct/fusion/ml"
            }
        )
    
    try:
        # 解码 Base64
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        try:
            image_bytes = base64.b64decode(image_base64)
        except Exception:
            logger.warning(f"[{request_id}] Base64 解码失败")
            return JSONResponse(
                status_code=400,
                content={
                    "status": "0002:base64解码异常.",
                    "message": "image_base64参数解码失败"
                }
            )
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            logger.warning(f"[{request_id}] 无法解析图片")
            return JSONResponse(
                status_code=400,
                content={
                    "status": "0007:请求参数内容错误或为空.",
                    "message": "无法解析图片，请确保是有效的图片格式"
                }
            )
        
        logger.debug(f"[{request_id}] 图片尺寸: {image.shape}")
        
        # 保存临时文件
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            cv2.imwrite(tmp.name, image)
            tmp_path = tmp.name
        
        try:
            result = {
                "is_tampered": False,
                "confidence": 0.0,
                "gb_confidence": None,
                "mask_base64": None,
                "marked_image_base64": None,
                "tamper_regions": None,
                "algorithm": algorithm,
                "processing_time": 0.0,
                "status": "0001:解析成功."
            }
            
            # 1. GB 分类器前置检测
            if not skip_gb:
                gb_result = gb_classify(tmp_path)
                result['gb_confidence'] = gb_result['confidence']
                
                # 如果 GB 判断为正常，直接返回
                if not gb_result['is_tampered']:
                    result['is_tampered'] = False
                    result['confidence'] = gb_result['confidence']
                    result['processing_time'] = time.time() - start_time
                    logger.info(f"[{request_id}] GB 判断为正常，跳过区域定位")
                    return result
            
            # 2. 如果篡改，进行区域定位
            if algorithm == 'ml':
                detect_result = _detect_with_ml(tmp_path, image)
            elif algorithm == 'ela':
                detect_result = _detect_with_ela(image)
            elif algorithm == 'dct':
                detect_result = _detect_with_dct(image)
            elif algorithm == 'fusion':
                detect_result = _detect_with_fusion(image)
            else:
                detect_result = {'is_tampered': True, 'confidence': 1.0, 'mask': None}
            
            # 合并结果
            result['is_tampered'] = detect_result['is_tampered']
            result['confidence'] = detect_result['confidence']
            result['mask_base64'] = detect_result.get('mask_base64')
            result['marked_image_base64'] = detect_result.get('marked_image_base64')
            result['tamper_regions'] = detect_result.get('tamper_regions')
            
            # 计算处理时间
            result['processing_time'] = time.time() - start_time
            
            logger.info(f"[{request_id}] 检测完成: is_tampered={result['is_tampered']}, "
                       f"confidence={result['confidence']:.4f}, time={result['processing_time']*1000:.1f}ms")
            
            return result
        
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    except Exception as e:
        logger.error(f"[{request_id}] 检测异常: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "0000:未知异常.",
                "message": str(e)
            }
        )


def _detect_with_ml(image_path: str, original_image: np.ndarray) -> Dict:
    """使用 ML 检测器"""
    logger.debug("使用 ML 检测器")
    
    detector = get_ml_detector()
    result = detector.predict(image_path)
    
    # 确保转换为 Python 原生类型
    is_tampered = bool(result['is_tampered'])
    confidence = float(result['confidence'])
    mask = result.get('mask')
    
    # 调试信息
    if mask is not None:
        mask_sum = int(mask.sum())
        mask_nonzero = int(np.count_nonzero(mask))
        logger.debug(f"ML mask 统计: shape={mask.shape}, sum={mask_sum}, nonzero={mask_nonzero}")
    else:
        logger.debug("ML mask 为空")
    
    # 生成标记图片
    marked_image_base64 = None
    tamper_regions = None
    
    if mask is not None and mask.sum() > 0:
        # 创建标记图片 (在原图上绘制矩形框)
        marked_image_base64 = _create_marked_image(original_image, mask)
        # 提取篡改区域 - 使用模型配置的 min_area
        min_area = getattr(detector, 'min_area_threshold', 100)
        tamper_regions = _extract_regions(mask, min_area=min_area)
        logger.debug(f"ML 检测到篡改区域: {len(tamper_regions) if tamper_regions else 0} 个 (min_area={min_area})")
    
    return {
        "is_tampered": is_tampered,
        "confidence": confidence,
        "mask_base64": _mask_to_base64(mask),
        "marked_image_base64": marked_image_base64,
        "tamper_regions": tamper_regions
    }


def _detect_with_ela(image: np.ndarray) -> Dict:
    """使用 ELA 检测器"""
    logger.debug("使用 ELA 检测器")
    
    detector = get_ela_detector()
    
    # 生成热力图
    heatmap = detector.detect(image)
    
    # 生成掩码
    mask = detector.get_mask(heatmap, method='otsu')
    
    # 计算篡改比例作为置信度 (转换为 Python 原生类型)
    tamper_ratio = float(np.sum(mask > 0)) / mask.size
    is_tampered = bool(tamper_ratio > 0.01)  # 超过1%像素认为篡改
    confidence = float(min(tamper_ratio * 10, 1.0))  # 归一化置信度
    
    logger.debug(f"ELA 检测: tamper_ratio={tamper_ratio:.4f}, is_tampered={is_tampered}")
    
    # 创建标记图片
    marked_image_base64 = None
    tamper_regions = None
    
    if mask is not None and mask.sum() > 0:
        marked_image_base64 = _create_marked_image(image, mask)
        tamper_regions = _extract_regions(mask)
    
    return {
        "is_tampered": is_tampered,
        "confidence": confidence,
        "mask_base64": _mask_to_base64(mask),
        "marked_image_base64": marked_image_base64,
        "tamper_regions": tamper_regions
    }


def _detect_with_dct(image: np.ndarray) -> Dict:
    """使用 DCT 检测器"""
    logger.debug("使用 DCT 检测器")
    
    detector = get_dct_detector()
    
    # 生成热力图
    heatmap = detector.detect(image)
    
    # 生成掩码
    mask = detector.get_mask(heatmap)
    
    # 计算篡改比例作为置信度 (转换为 Python 原生类型)
    tamper_ratio = float(np.sum(mask > 0)) / mask.size
    is_tampered = bool(tamper_ratio > 0.01)
    confidence = float(min(tamper_ratio * 10, 1.0))
    
    logger.debug(f"DCT 检测: tamper_ratio={tamper_ratio:.4f}, is_tampered={is_tampered}")
    
    # 创建标记图片
    marked_image_base64 = None
    tamper_regions = None
    
    if mask is not None and mask.sum() > 0:
        marked_image_base64 = _create_marked_image(image, mask)
        tamper_regions = _extract_regions(mask)
    
    return {
        "is_tampered": is_tampered,
        "confidence": confidence,
        "mask_base64": _mask_to_base64(mask),
        "marked_image_base64": marked_image_base64,
        "tamper_regions": tamper_regions
    }


def _detect_with_fusion(image: np.ndarray) -> Dict:
    """使用融合检测器"""
    logger.debug("使用融合检测器")
    
    ela_detector = get_ela_detector()
    dct_detector = get_dct_detector()
    fusion = get_fusion_detector()
    
    # ELA 检测
    ela_heatmap = ela_detector.detect(image)
    ela_mask = ela_detector.get_mask(ela_heatmap, method='otsu')
    
    # DCT 检测
    dct_heatmap = dct_detector.detect(image)
    dct_mask = dct_detector.get_mask(dct_heatmap)
    
    # 融合热力图
    heatmaps = {
        'ela': ela_heatmap,
        'dct': dct_heatmap
    }
    fused_heatmap = fusion.fusion_adaptive(heatmaps, is_jpeg=True)
    
    # 生成融合掩码
    mask = fusion.threshold(fused_heatmap, method='fixed', threshold=0.2)
    
    # 计算篡改比例作为置信度 (转换为 Python 原生类型)
    tamper_ratio = float(np.sum(mask > 0)) / mask.size
    is_tampered = bool(tamper_ratio > 0.01)
    confidence = float(min(tamper_ratio * 10, 1.0))
    
    logger.debug(f"Fusion 检测: tamper_ratio={tamper_ratio:.4f}, is_tampered={is_tampered}")
    
    # 创建标记图片
    marked_image_base64 = None
    tamper_regions = None
    
    if mask is not None and mask.sum() > 0:
        marked_image_base64 = _create_marked_image(image, mask)
        tamper_regions = _extract_regions(mask)
    
    return {
        "is_tampered": is_tampered,
        "confidence": confidence,
        "mask_base64": _mask_to_base64(mask),
        "marked_image_base64": marked_image_base64,
        "tamper_regions": tamper_regions
    }


def _extract_regions(mask: np.ndarray, min_area: int = 100) -> Optional[List[TamperRegion]]:
    """
    从掩码中提取篡改区域矩形坐标
    
    Args:
        mask: 二值掩码 (0=正常, 255=篡改)
        min_area: 最小区域面积阈值
    
    Returns:
        篡改区域列表，如果无篡改则返回 null
    """
    if mask is None:
        logger.debug(f"掩码为空，无法提取区域")
        return None
    
    mask_sum = int(mask.sum())
    if mask_sum == 0:
        logger.debug(f"掩码全为0，无篡改区域")
        return None
    
    logger.debug(f"掩码非零像素数: {mask_sum}")
    
    # 确保是二值图
    if len(mask.shape) == 3:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask
    
    _, mask_binary = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    
    # 查找轮廓
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(contour)
            regions.append(TamperRegion(
                left_top=[int(x), int(y)],
                right_bottom=[int(x + w), int(y + h)]
            ))
    
    return regions if regions else None


def _mask_to_base64(mask: np.ndarray) -> Optional[str]:
    """掩码转 Base64"""
    if mask is None:
        return None
    
    _, buffer = cv2.imencode('.png', mask)
    return base64.b64encode(buffer).decode('utf-8')


def _create_marked_image(image: np.ndarray, mask: np.ndarray) -> Optional[str]:
    """
    在原图上绘制篡改区域矩形框
    
    Args:
        image: 原图 (BGR格式)
        mask: 篡改掩码 (二值图，255为篡改区域)
    
    Returns:
        标记后的图片 Base64 编码
    """
    if mask is None or int(mask.sum()) == 0:
        return None
    
    # 确保掩码和原图尺寸一致
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    
    marked_image = image.copy()
    
    # 查找轮廓
    if len(mask.shape) == 3:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask
    
    _, mask_binary = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 绘制红色轮廓
    cv2.drawContours(marked_image, contours, -1, (0, 0, 255), 2)
    
    # 绘制矩形框
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= 100:
            x, y, w, h = cv2.boundingRect(contour)
            # 绿色矩形框
            cv2.rectangle(marked_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # 编码为 Base64
    _, buffer = cv2.imencode('.jpg', marked_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return base64.b64encode(buffer).decode('utf-8')


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='图像篡改检测服务')
    parser.add_argument('--port', type=int, default=8000, help='服务端口')
    
    args = parser.parse_args()
    
    # 配置日志 (自动保存到 ./logs，每30天清理)
    setup_logging()
    
    # 启动时加载 GB 分类器
    load_gb_classifier()
    
    host = '0.0.0.0'
    workers = 8  # 默认8并发
    
    logger.info(f"启动服务: http://{host}:{args.port}")
    logger.info(f"Workers: {workers}")
    
    # 启动服务
    uvicorn.run(
        "server_forgrey:app",
        host=host,
        port=args.port,
        workers=workers
    )


if __name__ == '__main__':
    main()