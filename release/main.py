#!/usr/bin/env python3
"""
图像篡改检测服务

FastAPI 服务，提供 REST API 接口

入参:
- image_base64: 图片 Base64 编码
- algorithm: 算法名称 (ela/dct/fusion/ml)

出参:
- is_tampered: 是否篡改
- confidence: 置信度
- mask_base64: 篡改区域掩码 Base64
- marked_image_base64: 原图标记篡改区域图片 Base64
"""

import os
import sys
import base64
import tempfile
import uuid
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 导入检测器
from algorithms import ELADetector, DCTDetector, FusionDetector
from algorithms.ml_detector import MLDetectorGPU

# 创建 FastAPI 应用
app = FastAPI(
    title="图像篡改检测服务",
    description="检测图像是否经过拼接、复制粘贴、修饰等篡改操作",
    version="1.0.0"
)

# 初始化检测器
detectors = {}


def get_detector(algorithm: str):
    """获取检测器实例"""
    if algorithm in detectors:
        return detectors[algorithm]
    
    if algorithm == 'ela':
        detectors['ela'] = ELADetector()
        return detectors['ela']
    
    elif algorithm == 'dct':
        detectors['dct'] = DCTDetector()
        return detectors['dct']
    
    elif algorithm == 'fusion':
        detectors['fusion'] = FusionDetector()
        return detectors['fusion']
    
    elif algorithm == 'ml':
        detectors['ml'] = MLDetectorGPU()
        return detectors['ml']
    
    else:
        raise HTTPException(status_code=400, detail=f"未知算法: {algorithm}")


# 响应模型
class DetectionResponse(BaseModel):
    """检测响应"""
    is_tampered: bool              # 是否篡改
    confidence: float              # 置信度 (0-1)
    mask_base64: Optional[str]     # 篡改区域掩码 Base64
    marked_image_base64: Optional[str]  # 原图标记篡改区域图片 Base64
    algorithm: str                 # 使用的算法
    processing_time: float         # 处理时间 (秒)


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    timestamp: str
    algorithms: list


# API 端点
@app.get("/", response_model=HealthResponse)
async def root():
    """根路径 - 健康检查"""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "algorithms": ["ela", "dct", "fusion", "ml"]
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """健康检查"""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "algorithms": ["ela", "dct", "fusion", "ml"]
    }


@app.post("/tamper_detection/v1/tamper_detect_img", response_model=DetectionResponse)
async def tamper_detect_img(
    image_base64: str = Form(..., description="图片 Base64 编码"),
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
    
    try:
        # 解码 Base64
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        image_bytes = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="无法解析图片")
        
        # 保存临时文件
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            cv2.imwrite(tmp.name, image)
            tmp_path = tmp.name
        
        try:
            detector = get_detector(algorithm)
            
            if algorithm == 'ml':
                result = detector.predict(tmp_path)
                is_tampered = result['is_tampered']
                confidence = result['confidence']
                mask_base64 = result.get('mask_base64')
                mask = result.get('mask')
            
            elif algorithm in ['ela', 'dct', 'fusion']:
                result = detector.detect(tmp_path)
                is_tampered = result.get('is_tampered', False)
                confidence = result.get('confidence', 0.0)
                mask = result.get('mask')
                mask_base64 = _mask_to_base64(mask)
            
            else:
                raise HTTPException(status_code=400, detail=f"未知算法: {algorithm}")
            
            # 生成标记图片
            marked_image_base64 = _create_marked_image(image, mask)
            
            processing_time = time.time() - start_time
            
            return {
                "is_tampered": is_tampered,
                "confidence": confidence,
                "mask_base64": mask_base64,
                "marked_image_base64": marked_image_base64,
                "algorithm": algorithm,
                "processing_time": processing_time
            }
        
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    if mask is None or mask.sum() == 0:
        return None
    
    # 确保掩码和原图尺寸一致
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    
    # 创建红色半透明覆盖层
    marked_image = image.copy()
    
    # 方法：在篡改区域绘制红色轮廓
    # 查找轮廓
    if len(mask.shape) == 3:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask
    
    # 二值化确保是二值图
    _, mask_binary = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 绘制红色轮廓（线宽3）
    cv2.drawContours(marked_image, contours, -1, (0, 0, 255), 3)
    
    # 在篡改区域填充半透明红色
    overlay = marked_image.copy()
    cv2.fillPoly(overlay, contours, (0, 0, 255))
    cv2.addWeighted(overlay, 0.3, marked_image, 0.7, 0, marked_image)
    
    # 编码为 Base64
    _, buffer = cv2.imencode('.jpg', marked_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return base64.b64encode(buffer).decode('utf-8')


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)