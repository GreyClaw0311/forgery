"""
Pydantic 数据模型定义
"""

from pydantic import BaseModel, Field
from typing import Optional


class DetectionRequest(BaseModel):
    """检测请求模型"""
    image_base64: str = Field(..., description="Base64编码的图像数据")
    algorithm: str = Field(..., description="检测算法名称")


class DetectionResponse(BaseModel):
    """检测响应模型"""
    is_tampered: bool = Field(..., description="是否检测到篡改")
    confidence: float = Field(..., description="置信度 (0-1)", ge=0.0, le=1.0)
    mask_image: str = Field(..., description="检测结果图片 (Base64编码)")
    algorithm: str = Field(..., description="使用的算法名称")
    tampered_ratio: float = Field(0.0, description="篡改区域占比", ge=0.0, le=1.0)


class AlgorithmInfo(BaseModel):
    """算法信息模型"""
    name: str
    description: str
    suitable: list[str]
    speed: str
    accuracy: Optional[str] = None


class ServiceStatus(BaseModel):
    """服务状态模型"""
    service: str
    version: str
    status: str
    algorithms: list[str]