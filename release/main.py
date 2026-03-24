#!/usr/bin/env python3
"""
图像篡改检测服务

FastAPI 服务，提供 REST API 接口

入参:
- image_base64: 图片 Base64 编码
- algorithm: 算法名称 (ela/dct/fusion/ml)

出参:
- status: 状态码
- is_tampered: 是否篡改
- confidence: 置信度
- mask_base64: 篡改区域掩码 Base64
- marked_image_base64: 原图标记篡改区域图片 Base64
- tamper_regions: 篡改区域矩形坐标列表
"""

import os
import sys
import base64
import tempfile
import uuid
from datetime import datetime
from typing import Optional, List, Dict

import cv2
import numpy as np

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 导入检测器
from algorithms.ela_detector import ELADetector
from algorithms.dct_detector import DCTBlockDetector
from algorithms.fusion_detector import AdaptiveFusion
from algorithms.ml_detector import MLDetectorGPU

# 状态码定义
STATUS_CODES = {
    "0000": "未知异常.",
    "0001": "解析成功.",
    "0002": "base64解码异常.",
    "0004": "参数格式错误.",
    "0007": "请求参数内容错误或为空."
}

# 创建 FastAPI 应用
app = FastAPI(
    title="图像篡改检测服务",
    description="检测图像是否经过拼接、复制粘贴、修饰等篡改操作",
    version="1.0.0"
)

# 初始化检测器
detectors = {}


def get_ela_detector():
    """获取 ELA 检测器"""
    if 'ela' not in detectors:
        detectors['ela'] = ELADetector()
    return detectors['ela']


def get_dct_detector():
    """获取 DCT 检测器"""
    if 'dct' not in detectors:
        detectors['dct'] = DCTBlockDetector()
    return detectors['dct']


def get_fusion_detector():
    """获取融合检测器"""
    if 'fusion' not in detectors:
        detectors['fusion'] = AdaptiveFusion()
    return detectors['fusion']


def get_ml_detector():
    """获取 ML 检测器"""
    if 'ml' not in detectors:
        detectors['ml'] = MLDetectorGPU()
    return detectors['ml']


# 响应模型
class TamperRegion(BaseModel):
    """篡改区域矩形"""
    left_top: List[int]      # 左上角坐标 [x, y]
    right_bottom: List[int]  # 右下角坐标 [x, y]


class DetectionResponse(BaseModel):
    """检测响应"""
    status: str                          # 状态码
    is_tampered: bool                    # 是否篡改
    confidence: float                    # 置信度 (0-1)
    mask_base64: Optional[str]           # 篡改区域掩码 Base64
    marked_image_base64: Optional[str]   # 原图标记篡改区域图片 Base64
    tamper_regions: Optional[List[TamperRegion]]  # 篡改区域矩形坐标列表
    algorithm: str                       # 使用的算法
    processing_time: float               # 处理时间 (秒)


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    timestamp: str
    algorithms: list


class ErrorResponse(BaseModel):
    """错误响应"""
    status: str
    message: str


# 全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理 - 返回0000状态码"""
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
        "algorithms": ["ela", "dct", "fusion", "ml"]
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """健康检查"""
    return {
        "status": "0001:服务运行正常.",
        "timestamp": datetime.now().isoformat(),
        "algorithms": ["ela", "dct", "fusion", "ml"]
    }


@app.post("/tamper_detection/v1/tamper_detect_img")
async def tamper_detect_img(
    image_base64: str = Form(default="", description="图片 Base64 编码"),
    algorithm: str = Form(default="ml", description="算法名称: ela/dct/fusion/ml")
):
    """
    检测图片是否篡改
    
    - **image_base64**: 图片 Base64 编码，支持带或不带 data URI 前缀
    - **algorithm**: 算法名称
        - ela: ELA 单特征检测
        - dct: DCT 单特征检测
        - fusion: 多特征融合检测
        - ml: 机器学习串联检测 (推荐)
    """
    import time
    
    start_time = time.time()
    
    # 检查参数是否为空
    if not image_base64 or image_base64.strip() == "":
        return JSONResponse(
            status_code=400,
            content={
                "status": "0007:请求参数内容错误或为空.",
                "message": "image_base64参数不能为空"
            }
        )
    
    # 检查算法参数
    if algorithm not in ['ela', 'dct', 'fusion', 'ml']:
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
            return JSONResponse(
                status_code=400,
                content={
                    "status": "0007:请求参数内容错误或为空.",
                    "message": "无法解析图片，请确保是有效的图片格式"
                }
            )
        
        # 保存临时文件 (ML 检测器需要文件路径)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            cv2.imwrite(tmp.name, image)
            tmp_path = tmp.name
        
        try:
            # 根据算法调用不同检测器
            if algorithm == 'ml':
                result = _detect_with_ml(tmp_path, image)
            elif algorithm == 'ela':
                result = _detect_with_ela(image)
            elif algorithm == 'dct':
                result = _detect_with_dct(image)
            elif algorithm == 'fusion':
                result = _detect_with_fusion(image)
            
            # 计算处理时间
            result['processing_time'] = time.time() - start_time
            result['algorithm'] = algorithm
            result['status'] = "0001:解析成功."
            
            return result
        
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "0000:未知异常.",
                "message": str(e)
            }
        )


def _detect_with_ml(image_path: str, original_image: np.ndarray) -> Dict:
    """使用 ML 检测器"""
    detector = get_ml_detector()
    result = detector.predict(image_path)
    
    # 确保转换为 Python 原生类型
    is_tampered = bool(result['is_tampered'])
    confidence = float(result['confidence'])
    mask = result.get('mask')
    
    return {
        "is_tampered": is_tampered,
        "confidence": confidence,
        "mask_base64": _mask_to_base64(mask),
        "marked_image_base64": _create_marked_image(original_image, mask),
        "tamper_regions": _extract_regions(mask)
    }


def _detect_with_ela(image: np.ndarray) -> Dict:
    """使用 ELA 检测器"""
    detector = get_ela_detector()
    
    # 生成热力图
    heatmap = detector.detect(image)
    
    # 生成掩码
    mask = detector.get_mask(heatmap, method='otsu')
    
    # 计算篡改比例作为置信度
    tamper_ratio = float(np.sum(mask > 0)) / mask.size
    is_tampered = bool(tamper_ratio > 0.01)  # 超过1%像素认为篡改
    confidence = float(min(tamper_ratio * 10, 1.0))  # 归一化置信度
    
    return {
        "is_tampered": is_tampered,
        "confidence": confidence,
        "mask_base64": _mask_to_base64(mask),
        "marked_image_base64": _create_marked_image(image, mask),
        "tamper_regions": _extract_regions(mask)
    }


def _detect_with_dct(image: np.ndarray) -> Dict:
    """使用 DCT 检测器"""
    detector = get_dct_detector()
    
    # 生成热力图
    heatmap = detector.detect(image)
    
    # 生成掩码
    mask = detector.get_mask(heatmap)
    
    # 计算篡改比例作为置信度 (转换为 Python 原生类型)
    tamper_ratio = float(np.sum(mask > 0)) / mask.size
    is_tampered = bool(tamper_ratio > 0.01)
    confidence = float(min(tamper_ratio * 10, 1.0))
    
    return {
        "is_tampered": is_tampered,
        "confidence": confidence,
        "mask_base64": _mask_to_base64(mask),
        "marked_image_base64": _create_marked_image(image, mask),
        "tamper_regions": _extract_regions(mask)
    }


def _detect_with_fusion(image: np.ndarray) -> Dict:
    """使用融合检测器"""
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
    
    return {
        "is_tampered": is_tampered,
        "confidence": confidence,
        "mask_base64": _mask_to_base64(mask),
        "marked_image_base64": _create_marked_image(image, mask),
        "tamper_regions": _extract_regions(mask)
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
    if mask is None or int(mask.sum()) == 0:
        return None
    
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
    在原图上标记篡改区域
    
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
    cv2.drawContours(marked_image, contours, -1, (0, 0, 255), 3)
    
    # 填充半透明红色
    overlay = marked_image.copy()
    cv2.fillPoly(overlay, contours, (0, 0, 255))
    cv2.addWeighted(overlay, 0.3, marked_image, 0.7, 0, marked_image)
    
    # 绘制矩形框
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= 100:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(marked_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # 编码为 Base64
    _, buffer = cv2.imencode('.jpg', marked_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return base64.b64encode(buffer).decode('utf-8')


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)