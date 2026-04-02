# Forgrey 图像篡改检测服务 Docker 镜像
# 构建时间: 2026-04-02
# 基础镜像: ontology_semantic-debian12-amd64-py3.10.13-cuda12.8-dev-wangyi-develop:v1.0.0
# Python 版本: 3.11.14 (从基础镜像升级)

# ==================== 基础镜像 ====================
FROM ontology_semantic-debian12-amd64-py3.10.13-cuda12.8-dev-wangyi-develop:v1.0.0

# ==================== 元信息 ====================
LABEL maintainer="灰 (上坤商业帝国首席CTO)"
LABEL version="1.1.0"
LABEL description="Forgrey 图像篡改检测服务 - GB分类器 + 像素级定位 - Python 3.11.14"

# ==================== 环境变量 ====================
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Shanghai
ENV PYTHON_VERSION=3.11.14
ENV PYPI_MIRROR=https://mirrors.aliyun.com/pypi/simple/

# ==================== 国内镜像源配置 ====================
# 使用阿里云镜像加速
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources && \
    sed -i 's/security.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources

# ==================== 安装编译依赖 ====================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    curl \
    ca-certificates \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    libgdbm-dev \
    libdb-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ==================== 安装 Python 3.11.14 ====================
# 从淘宝镜像下载 Python 源码 (国内加速)
RUN wget -q https://mirrors.tuna.tsinghua.edu.cn/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tar.xz \
    -O /tmp/Python-${PYTHON_VERSION}.tar.xz && \
    cd /tmp && \
    tar -xf Python-${PYTHON_VERSION}.tar.xz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure \
        --prefix=/usr/local/python3.11 \
        --enable-optimizations \
        --with-lto \
        --enable-shared \
        --with-system-ffi \
        --with-ensurepip=install \
        LDFLAGS="-Wl,-rpath,/usr/local/python3.11/lib" && \
    make -j$(nproc) && \
    make install && \
    cd / && \
    rm -rf /tmp/Python-${PYTHON_VERSION}* && \
    ln -sf /usr/local/python3.11/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/local/python3.11/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/local/python3.11/bin/pip3.11 /usr/bin/pip3 && \
    ln -sf /usr/local/python3.11/bin/pip3.11 /usr/bin/pip && \
    ldconfig

# ==================== 配置 pip 镜像源 ====================
# 使用阿里云 pip 镜像 (国内加速)
RUN pip config set global.index-url ${PYPI_MIRROR} && \
    pip config set global.trusted-host mirrors.aliyun.com && \
    pip install --upgrade pip setuptools wheel

# ==================== 工作目录 ====================
WORKDIR /app/forgrey

# ==================== 复制文件 ====================
# 复制 release 目录到镜像
COPY release/ /app/forgrey/

# ==================== 安装 Python 依赖 ====================
# 使用国内镜像安装依赖
RUN pip install --no-cache-dir -r requirements.txt && \
    echo "✓ 依赖安装完成"

# ==================== 创建必要目录 ====================
# 创建日志目录
RUN mkdir -p /app/forgrey/logs && \
    chmod +x /app/forgrey/start.sh

# ==================== 验证 Python 版本 ====================
RUN python --version && echo "✓ Python 版本验证完成"

# ==================== 模型文件 ====================
# 注意: 模型文件需要在运行时挂载或预先复制
# 模型目录结构:
#   models/gb_classifier/model.pkl
#   models/gb_classifier/scaler.pkl
#   models/gb_classifier/config.json
#   models/pixel_segmentation/model.pkl

# ==================== 端口暴露 ====================
EXPOSE 8000

# ==================== 健康检查 ====================
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ==================== 启动命令 ====================
# 前台运行服务 (-f 参数)
CMD ["bash", "start.sh", "-f"]

# ==================== 使用说明 ====================
# 
# 构建镜像:
#   docker build -t forgrey:v1.1.0 .
#
# 运行容器 (无模型):
#   docker run -d -p 8000:8000 --name forgrey forgrey:v1.1.0
#
# 运行容器 (挂载模型):
#   docker run -d -p 8000:8000 \
#     -v /path/to/models:/app/forgrey/models \
#     --name forgrey forgrey:v1.1.0
#
# 运行容器 (GPU 支持):
#   docker run -d -p 8000:8000 \
#     --gpus all \
#     -v /path/to/models:/app/forgrey/models \
#     --name forgrey forgrey:v1.1.0
#
# 查看日志:
#   docker logs -f forgrey
#
# 停止容器:
#   docker stop forgrey
#
# ==================== 注意事项 ====================
# 1. Python 3.11.14 从清华镜像下载源码编译安装
# 2. pip 使用阿里云镜像加速
# 3. apt 使用阿里云镜像加速
# 4. 模型文件较大，建议通过 volume 挂载
# 5. GPU 支持需要 nvidia-docker 或 docker --gpus
# 6. 默认端口 8000，可通过 -p 映射
# 7. 内存建议 8GB+，像素级检测需要较大内存
# 8. 编译时间较长 (约 10-15 分钟)，建议使用 BuildKit 缓存