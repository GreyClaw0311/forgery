FROM python:3.9-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
RUN pip install --no-cache-dir \
    opencv-python==4.8.1.78 \
    numpy==1.24.3 \
    scikit-learn==1.3.2 \
    scipy==1.11.4 \
    matplotlib==3.8.2 \
    pillow==10.1.0 \
    pandas==2.1.3

# 拷贝项目代码
COPY . /app

CMD ["/bin/bash"]
