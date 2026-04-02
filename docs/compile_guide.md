# Python 代码加密编译指南

## 概述

将 `release/` 目录下的 Python 代码编译成 `.so` 文件（二进制加密），保护源代码。

## 输出

- `release_so/` 目录：编译后的代码 + 模型 + 启动脚本
- 所有 `.py` 文件编译为 `.so` 文件（无法直接阅读源码）

## 使用步骤

### 1. 安装依赖

```bash
pip install cython numpy
apt-get install gcc  # Linux 系统自带
```

### 2. 运行编译脚本

```bash
cd /root/forgery
python scripts/compile_to_so.py
```

### 3. 验证输出

```bash
# 检查 .so 文件
ls -la release_so/algorithms/*.so

# 测试导入
cd release_so
python test_import.py
```

### 4. 启动服务

```bash
cd release_so
bash start.sh
```

## 编译文件列表

| 文件 | 编译后 | 说明 |
|------|--------|------|
| server_forgrey.py | server_forgrey.so | 主服务 |
| algorithms/*.py | algorithms/*.so | 检测算法 |
| utils/*.py | utils/*.so | 工具函数 |
| test_service.py | test_service.so | 测试脚本 |

## 复制文件（不编译）

| 文件 | 说明 |
|------|------|
| requirements.txt | 依赖配置 |
| start.sh | 启动脚本 |
| stop.sh | 停止脚本 |
| models/ | 模型目录（运行时挂载） |

## 注意事项

1. **必须在 Linux 系统运行**：`.so` 文件是 Linux 动态库
2. **Python 版本一致**：编译环境的 Python 版本必须与运行环境一致
   - 编译: Python 3.10 → 运行: Python 3.10
   - 如果版本不同，需要重新编译
3. **模型文件处理**：
   - 编译时不包含模型（模型通过 Docker volume 挂载）
   - 确保 `models/gb_classifier/model.pkl` 等文件在运行时存在
4. **启动方式**：
   - 使用 `launcher.py` 启动（而非直接运行 server_forgrey.so）
   - `start.sh` 已自动更新为使用 launcher.py

## Docker 部署

编译后更新 Dockerfile：

```dockerfile
# 复制编译后的代码
COPY release_so/ /app/forgrey/
```

## 常见问题

### Q: 编译失败 "gcc not found"

安装 gcc：
```bash
apt-get install gcc g++
```

### Q: 导入失败 "cannot import name xxx"

检查 Python 版本是否一致：
```bash
# 编译环境
python --version

# 运行环境
docker exec -it forgrey python --version
```

### Q: 模型加载失败

确保模型文件存在：
```bash
ls release_so/models/gb_classifier/model.pkl
```

如果不存在，需要从训练服务器复制模型文件。

---

**作者**: 灰 (上坤商业帝国首席CTO)
**日期**: 2026-04-02