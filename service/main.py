#!/usr/bin/env python3
"""
Forgery Detection Service - 图像篡改检测服务

提供统一 API 接口，支持多种检测算法
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import base64
import numpy as np
import cv2
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from service.detector_wrapper import ForgeryDetectorWrapper
from service.models import DetectionRequest, DetectionResponse

# 创建 FastAPI 应用
app = FastAPI(
    title="Forgery Detection Service",
    description="图像篡改检测服务 - 支持多种检测算法",
    version="1.0.0"
)

# 初始化检测器
detector_wrapper = None


def get_detector() -> ForgeryDetectorWrapper:
    """获取检测器单例"""
    global detector_wrapper
    if detector_wrapper is None:
        detector_wrapper = ForgeryDetectorWrapper()
    return detector_wrapper


@app.on_event("startup")
async def startup_event():
    """启动时初始化"""
    print("正在初始化检测器...")
    get_detector()
    print("检测器初始化完成")


@app.get("/")
async def root():
    """服务状态"""
    return {
        "service": "Forgery Detection Service",
        "version": "1.0.0",
        "status": "running",
        "algorithms": [
            "ela",
            "dct", 
            "noise",
            "copy_move",
            "fusion",
            "pixel_ml",
            "pipeline"
        ]
    }


@app.get("/algorithms")
async def list_algorithms():
    """列出所有可用算法"""
    return {
        "algorithms": [
            {
                "name": "ela",
                "description": "ELA (Error Level Analysis) - JPEG压缩误差分析",
                "suitable": ["JPEG二次压缩", "图像拼接"],
                "speed": "fast"
            },
            {
                "name": "dct",
                "description": "DCT Block Detection - DCT块效应检测",
                "suitable": ["JPEG压缩不一致", "块边界篡改"],
                "speed": "fast"
            },
            {
                "name": "noise",
                "description": "Noise Consistency Detection - 噪声一致性分析",
                "suitable": ["拼接检测", "复制粘贴"],
                "speed": "medium"
            },
            {
                "name": "copy_move",
                "description": "Copy-Move Detection - 复制移动检测",
                "suitable": ["复制粘贴篡改", "区域克隆"],
                "speed": "medium"
            },
            {
                "name": "fusion",
                "description": "Multi-Detector Fusion - 多检测器融合",
                "suitable": ["综合篡改检测", "未知类型"],
                "speed": "slow"
            },
            {
                "name": "pixel_ml",
                "description": "Pixel-level ML - 像素级机器学习检测",
                "suitable": ["精确分割", "像素级定位"],
                "speed": "slow",
                "accuracy": "F1=0.898"
            },
            {
                "name": "pipeline",
                "description": "Full Pipeline - 完整检测流水线",
                "suitable": ["全面检测", "生产环境"],
                "speed": "slow"
            }
        ]
    }


@app.post("/detect", response_model=DetectionResponse)
async def detect_forgery(
    file: UploadFile = File(..., description="图像文件"),
    algorithm: str = Form(..., description="检测算法名称")
):
    """
    检测图像篡改
    
    - **file**: 上传的图像文件 (支持 JPG, PNG, BMP)
    - **algorithm**: 检测算法 (ela/dct/noise/copy_move/fusion/pixel_ml/pipeline)
    
    返回:
    - **is_tampered**: 是否检测到篡改
    - **confidence**: 置信度 (0-1)
    - **mask_image**: 检测结果图片 (Base64编码)
    """
    # 验证算法名称
    valid_algorithms = ["ela", "dct", "noise", "copy_move", "fusion", "pixel_ml", "pipeline"]
    if algorithm not in valid_algorithms:
        raise HTTPException(
            status_code=400, 
            detail=f"无效的算法名称: {algorithm}. 支持的算法: {valid_algorithms}"
        )
    
    # 读取图像
    try:
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="无法解析图像文件")
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图像读取失败: {str(e)}")
    
    # 执行检测
    try:
        detector = get_detector()
        result = detector.detect(image, algorithm)
        
        return DetectionResponse(
            is_tampered=result["is_tampered"],
            confidence=result["confidence"],
            mask_image=result["mask_base64"],
            algorithm=algorithm,
            tampered_ratio=result.get("tampered_ratio", 0.0)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检测失败: {str(e)}")


@app.post("/detect/base64", response_model=DetectionResponse)
async def detect_forgery_base64(request: DetectionRequest):
    """
    使用Base64编码图像检测篡改
    
    - **image_base64**: Base64编码的图像数据
    - **algorithm**: 检测算法名称
    
    返回检测结果
    """
    # 验证算法
    valid_algorithms = ["ela", "dct", "noise", "copy_move", "fusion", "pixel_ml", "pipeline"]
    if request.algorithm not in valid_algorithms:
        raise HTTPException(
            status_code=400,
            detail=f"无效的算法名称: {request.algorithm}. 支持的算法: {valid_algorithms}"
        )
    
    # 解码图像
    try:
        image_bytes = base64.b64decode(request.image_base64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="无法解析Base64图像数据")
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图像解码失败: {str(e)}")
    
    # 执行检测
    try:
        detector = get_detector()
        result = detector.detect(image, request.algorithm)
        
        return DetectionResponse(
            is_tampered=result["is_tampered"],
            confidence=result["confidence"],
            mask_image=result["mask_base64"],
            algorithm=request.algorithm,
            tampered_ratio=result.get("tampered_ratio", 0.0)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检测失败: {str(e)}")


@app.post("/detect/batch")
async def detect_batch(
    files: list[UploadFile] = File(...),
    algorithm: str = Form(...)
):
    """
    批量检测多张图像
    """
    results = []
    detector = get_detector()
    
    for file in files:
        try:
            image_bytes = await file.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                results.append({
                    "filename": file.filename,
                    "error": "无法解析图像"
                })
                continue
            
            result = detector.detect(image, algorithm)
            results.append({
                "filename": file.filename,
                "is_tampered": result["is_tampered"],
                "confidence": result["confidence"],
                "tampered_ratio": result.get("tampered_ratio", 0.0)
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)