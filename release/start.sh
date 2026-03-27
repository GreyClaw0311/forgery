#!/bin/bash
#
# 图像篡改检测服务启动脚本
#
# 用法:
#     ./start.sh          # 前台启动 (默认端口 8000)
#     ./start.sh -p 8080  # 指定端口启动
#     ./start.sh -d       # 后台启动 (守护进程模式)
#     ./start.sh -d -p 8080  # 后台启动并指定端口
#
# 功能:
#     1. 自动检查依赖
#     2. 自动检查端口占用
#     3. 自动创建日志目录
#     4. 支持前台/后台运行
#     5. 记录进程 PID
#

set -e

# 默认配置
PORT=8000
DAEMON=false
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PID_FILE="$SCRIPT_DIR/server.pid"
LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/server.log"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 帮助信息
usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -p, --port PORT    指定服务端口 (默认: 8000)"
    echo "  -d, --daemon       后台运行 (守护进程模式)"
    echo "  -h, --help         显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                 # 前台启动，端口 8000"
    echo "  $0 -p 8080         # 前台启动，端口 8080"
    echo "  $0 -d              # 后台启动，端口 8000"
    echo "  $0 -d -p 8080      # 后台启动，端口 8080"
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -d|--daemon)
            DAEMON=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}未知参数: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# 检查 Python
check_python() {
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}✗ Python3 未安装${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Python3 已安装: $(python3 --version)${NC}"
}

# 检查端口
check_port() {
    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${RED}✗ 端口 $PORT 已被占用${NC}"
        echo "占用进程:"
        lsof -i :$PORT
        exit 1
    fi
    echo -e "${GREEN}✓ 端口 $PORT 可用${NC}"
}

# 检查服务是否已运行
check_running() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p $PID > /dev/null 2>&1; then
            echo -e "${YELLOW}⚠ 服务已在运行 (PID: $PID)${NC}"
            echo "如需重启，请先运行: ./stop.sh"
            exit 1
        else
            # PID 文件存在但进程不存在，清理 PID 文件
            rm -f "$PID_FILE"
        fi
    fi
}

# 创建日志目录
create_log_dir() {
    if [ ! -d "$LOG_DIR" ]; then
        mkdir -p "$LOG_DIR"
        echo -e "${GREEN}✓ 创建日志目录: $LOG_DIR${NC}"
    fi
}

# 检查依赖
check_dependencies() {
    echo "检查依赖..."
    
    # 检查必要的 Python 包
    REQUIRED_PACKAGES=("fastapi" "uvicorn" "cv2" "numpy")
    MISSING_PACKAGES=()
    
    for pkg in "${REQUIRED_PACKAGES[@]}"; do
        if ! python3 -c "import $pkg" 2>/dev/null; then
            MISSING_PACKAGES+=("$pkg")
        fi
    done
    
    if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
        echo -e "${YELLOW}⚠ 缺少依赖包: ${MISSING_PACKAGES[*]}${NC}"
        echo "正在安装依赖..."
        pip install -r "$SCRIPT_DIR/requirements.txt" -q
        echo -e "${GREEN}✓ 依赖安装完成${NC}"
    else
        echo -e "${GREEN}✓ 依赖检查通过${NC}"
    fi
}

# 启动服务 (前台)
start_foreground() {
    echo "================================"
    echo "  图像篡改检测服务"
    echo "================================"
    echo "端口: $PORT"
    echo "模式: 前台运行"
    echo "日志: $LOG_FILE"
    echo "================================"
    echo ""
    echo "按 Ctrl+C 停止服务"
    echo ""
    
    cd "$SCRIPT_DIR"
    python3 server_forgrey.py --port $PORT
}

# 启动服务 (后台)
start_daemon() {
    echo "================================"
    echo "  图像篡改检测服务"
    echo "================================"
    echo "端口: $PORT"
    echo "模式: 后台运行 (守护进程)"
    echo "日志: $LOG_FILE"
    echo "================================"
    echo ""
    
    cd "$SCRIPT_DIR"
    nohup python3 server_forgrey.py --port $PORT >> "$LOG_FILE" 2>&1 &
    PID=$!
    echo $PID > "$PID_FILE"
    
    sleep 2
    
    if ps -p $PID > /dev/null 2>&1; then
        echo -e "${GREEN}✓ 服务启动成功 (PID: $PID)${NC}"
        echo ""
        echo "查看日志: tail -f $LOG_FILE"
        echo "停止服务: ./stop.sh"
    else
        echo -e "${RED}✗ 服务启动失败，请查看日志${NC}"
        cat "$LOG_FILE"
        rm -f "$PID_FILE"
        exit 1
    fi
}

# 主函数
main() {
    echo ""
    echo "启动图像篡改检测服务..."
    echo ""
    
    check_python
    check_port
    check_running
    create_log_dir
    check_dependencies
    
    echo ""
    
    if [ "$DAEMON" = true ]; then
        start_daemon
    else
        start_foreground
    fi
}

main