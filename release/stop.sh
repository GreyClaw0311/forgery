#!/bin/bash
#
# 图像篡改检测服务停止脚本
#
# 用法:
#     ./stop.sh           # 停止服务
#     ./stop.sh -f        # 强制停止 (如果正常停止失败)
#
# 功能:
#     1. 根据 PID 文件停止服务
#     2. 检查端口占用并停止
#     3. 清理 PID 文件
#

set -e

# 配置
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PID_FILE="$SCRIPT_DIR/server.pid"
DEFAULT_PORT=8000

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
    echo "  -f, --force    强制停止 (使用 SIGKILL)"
    echo "  -h, --help     显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0             # 停止服务"
    echo "  $0 -f          # 强制停止服务"
}

FORCE=false

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--force)
            FORCE=true
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

# 根据 PID 文件停止
stop_by_pid_file() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        echo "找到 PID 文件: $PID_FILE"
        echo "服务 PID: $PID"
        
        if ps -p $PID > /dev/null 2>&1; then
            echo "正在停止服务..."
            
            if [ "$FORCE" = true ]; then
                kill -9 $PID 2>/dev/null || true
            else
                kill $PID 2>/dev/null || true
                
                # 等待进程结束
                for i in {1..10}; do
                    if ! ps -p $PID > /dev/null 2>&1; then
                        break
                    fi
                    sleep 1
                done
                
                # 如果进程还在运行，强制停止
                if ps -p $PID > /dev/null 2>&1; then
                    echo -e "${YELLOW}服务未响应，强制停止...${NC}"
                    kill -9 $PID 2>/dev/null || true
                fi
            fi
            
            rm -f "$PID_FILE"
            echo -e "${GREEN}✓ 服务已停止${NC}"
            return 0
        else
            echo -e "${YELLOW}PID 文件存在但进程不存在，清理 PID 文件${NC}"
            rm -f "$PID_FILE"
        fi
    fi
    return 1
}

# 根据端口停止
stop_by_port() {
    echo "检查端口占用..."
    
    for PORT in 8000 8080; do
        PIDS=$(lsof -ti :$PORT 2>/dev/null || true)
        if [ -n "$PIDS" ]; then
            echo "发现端口 $PORT 被占用，PID: $PIDS"
            
            for PID in $PIDS; do
                echo "停止进程 $PID..."
                if [ "$FORCE" = true ]; then
                    kill -9 $PID 2>/dev/null || true
                else
                    kill $PID 2>/dev/null || true
                fi
            done
            
            sleep 1
            
            # 检查是否还有进程
            REMAINING=$(lsof -ti :$PORT 2>/dev/null || true)
            if [ -n "$REMAINING" ]; then
                echo -e "${YELLOW}强制停止剩余进程...${NC}"
                kill -9 $REMAINING 2>/dev/null || true
            fi
            
            echo -e "${GREEN}✓ 端口 $PORT 已释放${NC}"
            return 0
        fi
    done
    
    return 1
}

# 停止所有 Python server_forgrey.py 进程
stop_by_name() {
    echo "检查服务进程..."
    
    PIDS=$(pgrep -f "server_forgrey.py" 2>/dev/null || true)
    if [ -n "$PIDS" ]; then
        echo "发现服务进程: $PIDS"
        
        for PID in $PIDS; do
            echo "停止进程 $PID..."
            if [ "$FORCE" = true ]; then
                kill -9 $PID 2>/dev/null || true
            else
                kill $PID 2>/dev/null || true
            fi
        done
        
        sleep 1
        echo -e "${GREEN}✓ 服务进程已停止${NC}"
        return 0
    fi
    
    return 1
}

# 主函数
main() {
    echo ""
    echo "停止图像篡改检测服务..."
    echo ""
    
    STOPPED=false
    
    # 方式1: 根据 PID 文件停止
    if stop_by_pid_file; then
        STOPPED=true
    fi
    
    # 方式2: 根据端口停止
    if ! $STOPPED; then
        if stop_by_port; then
            STOPPED=true
        fi
    fi
    
    # 方式3: 根据进程名停止
    if ! $STOPPED; then
        if stop_by_name; then
            STOPPED=true
        fi
    fi
    
    # 结果
    echo ""
    if $STOPPED; then
        echo -e "${GREEN}✓ 服务停止完成${NC}"
    else
        echo -e "${YELLOW}⚠ 未发现运行中的服务${NC}"
    fi
    
    # 清理 PID 文件
    rm -f "$PID_FILE"
}

main