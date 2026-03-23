# 图像篡改检测服务接口文档

## 接口说明

针对上传的图像文件或Base64编码的图像数据进行篡改检测，识别图像是否经过拼接、复制粘贴、修饰等篡改操作，并返回篡改区域的掩码图像。该服务支持两种调用方式：通过multipart/form-data直接上传图像文件，或通过JSON请求体传递Base64编码的图像数据。

## 接口 URL

```
{服务IP}:{端口号}/detect
{服务IP}:{端口号}/detect_base64
```

## 访问方式

POST

---

## 接口入参说明

### 1. 文件上传接口 `/detect` (multipart/form-data)

| 参数名称 | 参数类型 | 是否必须 | 参数说明 |
|----------|----------|----------|----------|
| file | file | true | 要检测的图片文件，支持jpg/png格式 |
| algorithm | string | false | 算法名称，可选值：`ela`、`dct`、`fusion`、`ml`，默认值为`ml` |

**请求头需设置：** `Content-Type: multipart/form-data`

**请求示例 (curl):**

```bash
curl -X POST "http://localhost:8000/detect" \
  -F "file=@test_image.jpg" \
  -F "algorithm=ml"
```

### 2. Base64编码接口 `/detect_base64` (application/x-www-form-urlencoded)

| 参数名称 | 参数类型 | 是否必须 | 参数说明 |
|----------|----------|----------|----------|
| image_base64 | string | true | 图片的Base64编码字符串，支持带或不带data URI前缀 |
| algorithm | string | false | 算法名称，可选值：`ela`、`dct`、`fusion`、`ml`，默认值为`ml` |

**请求头需设置：** `Content-Type: application/x-www-form-urlencoded`

**请求示例:**

```json
{
  "image_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD...",
  "algorithm": "ml"
}
```

---

## 算法说明

| 算法名称 | 说明 | 推荐场景 |
|----------|------|----------|
| **ml** | 机器学习串联检测 (Random Forest) | **推荐**，综合检测效果最佳 |
| **fusion** | 多特征融合检测 | 需要快速初步筛查的场景 |
| **ela** | ELA (Error Level Analysis) 单特征检测 | JPEG重压缩检测 |
| **dct** | DCT (离散余弦变换) 单特征检测 | JPEG块效应检测 |

---

## 出参

接口返回统一的JSON格式响应。

**成功响应示例:**

```json
{
  "is_tampered": true,
  "confidence": 0.89,
  "mask_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "algorithm": "ml",
  "processing_time": 2.35
}
```

### 出参说明

| 参数名称 | 参数类型 | 是否必须 | 参数说明 |
|----------|----------|----------|----------|
| is_tampered | boolean | true | 是否检测到篡改，`true`表示篡改，`false`表示正常 |
| confidence | float | true | 置信度，取值范围0-1，值越高表示越确信 |
| mask_base64 | string | false | 篡改区域掩码图像的Base64编码(PNG格式)，白色区域表示篡改位置；若无篡改则为`null` |
| algorithm | string | true | 实际使用的算法名称 |
| processing_time | float | true | 处理耗时(秒) |

---

## 健康检查接口

### 接口 URL

```
{服务IP}:{端口号}/health
{服务IP}:{端口号}/
```

### 访问方式

GET

### 接口说明

用于检查服务是否正常运行，返回服务状态及可用算法列表。

### 出参示例

```json
{
  "status": "ok",
  "timestamp": "2026-03-20T18:20:00.123456",
  "algorithms": ["ela", "dct", "fusion", "ml"]
}
```

### 出参说明

| 参数名称 | 参数类型 | 是否必须 | 参数说明 |
|----------|----------|----------|----------|
| status | string | true | 服务状态，`ok`表示正常运行 |
| timestamp | string | true | 服务器时间戳(ISO 8601格式) |
| algorithms | list | true | 可用的检测算法列表 |

---

## 错误响应

当请求出错时，返回HTTP错误状态码及错误信息：

```json
{
  "detail": "错误描述信息"
}
```

### 常见错误码

| HTTP状态码 | 说明 |
|------------|------|
| 400 | 请求参数错误（不支持的文件类型、无效的Base64、未知的算法名称等） |
| 500 | 服务器内部错误 |

---

## 使用示例

### Python 示例

```python
import requests

# 文件上传方式
url = "http://localhost:8000/detect"
with open("test_image.jpg", "rb") as f:
    files = {"file": f}
    data = {"algorithm": "ml"}
    response = requests.post(url, files=files, data=data)
    result = response.json()
    print(f"是否篡改: {result['is_tampered']}")
    print(f"置信度: {result['confidence']}")

# Base64 方式
import base64
url = "http://localhost:8000/detect_base64"
with open("test_image.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode('utf-8')
    data = {"image_base64": image_base64, "algorithm": "ml"}
    response = requests.post(url, data=data)
    result = response.json()
```

### cURL 示例

```bash
# 文件上传
curl -X POST "http://localhost:8000/detect" \
  -F "file=@image.jpg" \
  -F "algorithm=ml"

# 健康检查
curl "http://localhost:8000/health"
```

---

## 服务启动

```bash
# 方式1: 直接运行
python main.py

# 方式2: 使用 uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000

# 方式3: Docker 部署
docker build -t forgery-detector .
docker run -p 8000:8000 forgery-detector
```

---

## 技术规格

| 项目 | 规格 |
|------|------|
| Web框架 | FastAPI |
| 图像处理 | OpenCV 4.x |
| 机器学习 | scikit-learn (Random Forest) |
| 支持格式 | JPEG, PNG |
| 默认端口 | 8000 |