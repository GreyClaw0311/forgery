# 图像篡改检测服务接口文档

## 接口说明

针对Base64编码的图像数据进行篡改检测，识别图像是否经过拼接、复制粘贴、修饰等篡改操作，并返回篡改区域的掩码图像、标记图片及矩形坐标。

## 接口 URL

```
{服务IP}:{端口号}/tamper_detection/v1/tamper_detect_img
```

## 访问方式

POST

---

## 接口入参说明

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
| **ml** | 机器学习串联检测 (Random Forest) | **推荐**，综合检测效果最佳，能输出精确篡改区域 |
| **fusion** | 多特征融合检测 (ELA + DCT) | 需要快速初步筛查的场景 |
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
  "marked_image_base64": "/9j/4AAQSkZJRgABAQAAAQABAAD...",
  "tamper_regions": [
    {
      "left": 120,
      "top": 85,
      "right": 276,
      "bottom": 227
    },
    {
      "left": 320,
      "top": 200,
      "right": 400,
      "bottom": 295
    }
  ],
  "algorithm": "ml",
  "processing_time": 2.35
}
```

### 出参说明

| 参数名称 | 参数类型 | 是否必须 | 参数说明 |
|----------|----------|----------|----------|
| is_tampered | boolean | true | 是否检测到篡改，`true`表示篡改，`false`表示正常 |
| confidence | float | true | 置信度，取值范围0-1，值越高表示越确信 |
| mask_base64 | string | false | 篡改区域掩码图像的Base64编码(PNG格式)，白色区域表示篡改位置；若无篡改或无法生成则为`null` |
| marked_image_base64 | string | false | 在原图上标记篡改区域的图片Base64编码(JPEG格式)，篡改区域用红色轮廓和半透明红色填充标注，绿色矩形框标出各篡改区域；若无篡改则为`null` |
| tamper_regions | array | false | 篡改区域矩形坐标列表，每个元素包含矩形位置和面积信息；若无篡改则为`null` |
| algorithm | string | true | 实际使用的算法名称 |
| processing_time | float | true | 处理耗时(秒) |

### tamper_regions 数组元素说明

| 参数名称 | 参数类型 | 说明 |
|----------|----------|------|
| left | int | 矩形左边界 x 坐标 (像素) |
| top | int | 矩形上边界 y 坐标 (像素) |
| right | int | 矩形右边界 x 坐标 (像素) |
| bottom | int | 矩形下边界 y 坐标 (像素) |

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
| 400 | 请求参数错误（无效的Base64、无法解析图片、未知的算法名称等） |
| 500 | 服务器内部错误 |

---

## 使用示例

### Python 示例

```python
import requests
import base64

# 读取图片并转为 Base64
url = "http://localhost:8000/tamper_detection/v1/tamper_detect_img"
with open("test_image.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode('utf-8')

# 发送请求
data = {
    "image_base64": image_base64,
    "algorithm": "ml"
}
response = requests.post(url, data=data)
result = response.json()

print(f"是否篡改: {result['is_tampered']}")
print(f"置信度: {result['confidence']}")

# 输出篡改区域坐标
if result['tamper_regions']:
    print("篡改区域:")
    for i, region in enumerate(result['tamper_regions'], 1):
        print(f"  区域{i}: 左({region['left']}) 上({region['top']}) 右({region['right']}) 下({region['bottom']})")

# 保存标记图片
if result['marked_image_base64']:
    marked_image = base64.b64decode(result['marked_image_base64'])
    with open("marked_result.jpg", "wb") as f:
        f.write(marked_image)

# 保存掩码图片
if result['mask_base64']:
    mask = base64.b64decode(result['mask_base64'])
    with open("mask_result.png", "wb") as f:
        f.write(mask)
```

### cURL 示例

```bash
# 篡改检测 (需要先转Base64)
IMAGE_BASE64=$(base64 -w 0 test_image.jpg)
curl -X POST "http://localhost:8000/tamper_detection/v1/tamper_detect_img" \
  -d "image_base64=$IMAGE_BASE64" \
  -d "algorithm=ml"

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