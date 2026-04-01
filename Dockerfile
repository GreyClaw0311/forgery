# Forgrey 图像篡改检测服务 Docker 镜像
# 构建时间: 2026-04-01
# 基础镜像: ontology_semantic-debian12-amd64-py3.10.13-cuda12.8-dev-wangyi-develop:v1.0.0

# ==================== 基础镜像 ====================
FROM ontology_semantic-debian12-amd64-py3.10.13-cuda12.8-dev-wangyi-develop:v1.0.0

# ==================== 元信息 ====================
LABEL maintainer="灰 (上坤商业帝国首席CTO)"
LABEL version="1.0.0"
LABEL description="Forgrey 图像篡改检测服务 - GB分类器 + 像素级定位"

# ==================== 环境变量 ====================
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Shanghai

# ==================== 工作目录 ====================
WORKDIR /app/forgrey

# ==================== 复制文件 ====================
# 复制 release 目录到镜像
COPY release/ /app/forgrey/

# ==================== 安装依赖 ====================
# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt && \
    echo "✓ 依赖安装完成"

# ==================== 创建必要目录 ====================
# 创建日志目录
RUN mkdir -p /app/forgrey/logs && \
    chmod +x /app/forgrey/start.sh

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
#   docker build -t forgrey:v1.0.0 .
#
# 运行容器 (无模型):
#   docker run -d -p 8000:8000 --name forgrey forgrey:v1.0.0
#
# 运行容器 (挂载模型):
#   docker run -d -p 8000:8000 \
#     -v /path/to/models:/app/forgrey/models \
#     --name forgrey forgrey:v1.0.0
#
# 运行容器 (GPU 支持):
#   docker run -d -p 8000:8000 \
#     --gpus all \
#     -v /path/to/models:/app/forgrey/models \
#     --name forgrey forgrey:v1.0.0
#
# 查看日志:
#   docker logs -f forgrey
#
# 停止容器:
#   docker stop forgrey
#
# ==================== 注意事项 ====================
# 1. 模型文件较大，建议通过 volume 挂载
# 2. GPU 支持需要 nvidia-docker 或 docker --gpus
# 3. 默认端口 8000，可通过 -p 映射
# 4. 内存建议 8GB+，像素级检测需要较大内存