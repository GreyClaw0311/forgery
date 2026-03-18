# Forgery Detection Service - API 文档

## 服务概述

图像篡改检测服务，提供统一 API 接口，支持多种检测算法。

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动服务

```bash
# 方式 1: 直接运行
python -m service.main

# 方式 2: 使用 uvicorn
uvicorn service.main:app --host 0.0.0.0 --port 8000

# 方式 3: 开发模式（热重载）
uvicorn service.main:app --reload --host 0.0.0.0 --port 8000
```

### 访问文档

服务启动后访问:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## API 接口

### 1. GET / - 服务状态

获取服务状态和可用算法列表。

**响应示例**:
```json
{
    "service": "Forgery Detection Service",
    "version": "1.0.0",
    "status": "running",
    "algorithms": ["ela", "dct", "noise", "copy_move", "fusion", "pixel_ml", "pipeline"]
}
```

---

### 2. GET /algorithms - 算法列表

获取所有可用算法的详细信息。

**响应示例**:
```json
{
    "algorithms": [
        {
            "name": "ela",
            "description": "ELA (Error Level Analysis) - JPEG压缩误差分析",
            "suitable": ["JPEG二次压缩", "图像拼接"],
            "speed": "fast"
        },
        ...
    ]
}
```

---

### 3. POST /detect - 检测图像篡改

**请求**: `multipart/form-data`

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| file | file | 是 | 图像文件 (JPG/PNG/BMP) |
| algorithm | string | 是 | 检测算法名称 |

**响应**:
```json
{
    "is_tampered": true,
    "confidence": 0.85,
    "mask_image": "base64_encoded_image...",
    "algorithm": "ela",
    "tampered_ratio": 0.12
}
```

**cURL 示例**:
```bash
curl -X POST "http://localhost:8000/detect" \
    -F "file=@/path/to/image.jpg" \
    -F "algorithm=ela"
```

---

### 4. POST /detect/base64 - Base64 图像检测

**请求**: `application/json`

```json
{
    "image_base64": "base64_encoded_image_data",
    "algorithm": "ela"
}
```

**响应**: 同 `/detect`

---

### 5. POST /detect/batch - 批量检测

**请求**: `multipart/form-data`

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| files | file[] | 是 | 多个图像文件 |
| algorithm | string | 是 | 检测算法名称 |

**响应**:
```json
{
    "results": [
        {
            "filename": "image1.jpg",
            "is_tampered": true,
            "confidence": 0.85,
            "tampered_ratio": 0.12
        },
        {
            "filename": "image2.jpg",
            "is_tampered": false,
            "confidence": 0.15,
            "tampered_ratio": 0.0
        }
    ]
}
```

---

## 算法说明

| 算法 | 描述 | 适用场景 | 速度 |
|------|------|----------|------|
| **ela** | JPEG压缩误差分析 | JPEG二次压缩、图像拼接 | 快 |
| **dct** | DCT块效应检测 | JPEG压缩不一致 | 快 |
| **noise** | 噪声一致性分析 | 拼接检测、复制粘贴 | 中 |
| **copy_move** | 复制移动检测 | 区域克隆、复制粘贴 | 中 |
| **fusion** | 多检测器融合 | 综合篡改检测 | 慢 |
| **pixel_ml** | 像素级机器学习 | 精确分割 (F1=0.898) | 慢 |
| **pipeline** | 完整检测流水线 | 生产环境 | 慢 |

---

## 返回字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| is_tampered | bool | 是否检测到篡改 |
| confidence | float | 置信度 (0-1) |
| mask_image | string | 检测结果图片 (Base64编码 PNG) |
| algorithm | string | 使用的算法名称 |
| tampered_ratio | float | 篡改区域占比 (0-1) |

---

## 错误响应

### 400 Bad Request
```json
{
    "detail": "无效的算法名称: xxx. 支持的算法: [...]"
}
```

### 500 Internal Server Error
```json
{
    "detail": "检测失败: xxx"
}
```

---

## Python 客户端示例

```python
import requests
import base64

# 方式 1: 上传文件
def detect_file(image_path: str, algorithm: str = "ela"):
    url = "http://localhost:8000/detect"
    
    with open(image_path, "rb") as f:
        files = {"file": f}
        data = {"algorithm": algorithm}
        response = requests.post(url, files=files, data=data)
    
    return response.json()

# 方式 2: Base64 编码
def detect_base64(image_path: str, algorithm: str = "ela"):
    url = "http://localhost:8000/detect/base64"
    
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode()
    
    payload = {
        "image_base64": image_base64,
        "algorithm": algorithm
    }
    response = requests.post(url, json=payload)
    
    return response.json()

# 使用示例
result = detect_file("test_image.jpg", algorithm="fusion")
print(f"是否篡改: {result['is_tampered']}")
print(f"置信度: {result['confidence']}")
print(f"篡改比例: {result['tampered_ratio']}")

# 保存结果图像
if result['mask_image']:
    mask_bytes = base64.b64decode(result['mask_image'])
    with open("result.png", "wb") as f:
        f.write(mask_bytes)
```

---

## Docker 部署

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "service.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# 构建镜像
docker build -t forgery-service .

# 运行容器
docker run -d -p 8000:8000 forgery-service
```

---

*文档版本: 2026-03-18*