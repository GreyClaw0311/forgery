# 图像篡改检测系统 Dockerfile
FROM python:3.9-slim

LABEL maintainer="GreyClaw <greyclaw@openclaw.ai>"
LABEL description="图像篡改检测系统"
LABEL version="1.0.0"

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY train/ ./train/
COPY release/ ./release/
COPY data/ ./data/

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

CMD ["python", "-m", "release.pipeline", "--help"]
