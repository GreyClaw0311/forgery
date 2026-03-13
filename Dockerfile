# 图像篡改检测系统 Dockerfile
# 基于 Python 3.9

FROM python:3.9-slim

LABEL maintainer="GreyClaw <greyclaw@openclaw.ai>"
LABEL description="图像篡改检测系统"
LABEL version="1.0.0"

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY src/ ./src/
COPY release/ ./release/
COPY train/ ./train/
COPY data/ ./data/

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# 默认命令
CMD ["python", "-m", "release.pipeline", "--help"]

# 使用示例:
# 构建镜像: docker build -t forgery-detector:1.0 .
# 运行检测: docker run --rm -v /path/to/images:/images forgery-detector:1.0 python -m release.pipeline /images/test.jpg