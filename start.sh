#!/usr/bin/env bash
set -euo pipefail

# 日志目录
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

echo "===== 1. 启动/重启 Milvus (Docker standalone) ====="
# stop 会在未运行时返回非零，忽略错误继续
bash milvus_standalone/standalone_embed.sh stop 2>/dev/null || true
bash milvus_standalone/standalone_embed.sh start

echo
 echo "===== 2. 启动 Embedding 服务 (port 8001) ====="
# 查占用 8001 端口的 PID 并 kill
EMB_PORT=8001
EMB_PIDS=$(lsof -t -i tcp:$EMB_PORT || true)
if [ -n "$EMB_PIDS" ]; then
  echo "Port $EMB_PORT 被占用，kill 进程: $EMB_PIDS"
  kill -9 $EMB_PIDS
else
  echo "Port $EMB_PORT 未被占用"
fi
# 后台启动，并将日志写入
nohup python src/api_embedding.py > "$LOG_DIR/api_embedding.log" 2>&1 &

echo
 echo "===== 3. 启动 Milvus API 服务 (port 8002) ====="
API_PORT=8002
API_PIDS=$(lsof -t -i tcp:$API_PORT || true)
if [ -n "$API_PIDS" ]; then
  echo "Port $API_PORT 被占用，kill 进程: $API_PIDS"
  kill -9 $API_PIDS
else
  echo "Port $API_PORT 未被占用"
fi
nohup python src/api_search_milvus.py > "$LOG_DIR/api_search_milvus.log" 2>&1 &

echo
 echo "===== 4. 启动 Streamlit 界面 ====="
# kill 之前可能已经跑的 streamlit_milvus.py
STREAMLIT_PIDS=$(pgrep -f "streamlit_milvus_search.py" || true)
if [ -n "$STREAMLIT_PIDS" ]; then
  echo "检测到已有 Streamlit 进程，kill: $STREAMLIT_PIDS"
  kill -9 $STREAMLIT_PIDS
else
  echo "未检测到已有 Streamlit 进程"
fi
nohup streamlit run src/streamlit_milvus_search.py > "$LOG_DIR/streamlit_milvus_search.log" 2>&1 &

echo
 echo "===== 5. 启动 Entity Cluster 界面 ====="
# kill 之前可能已经跑的 entity_cluster.py
ENTITY_PIDS=$(pgrep -f "streamlit_entity_cluster.py" || true)
if [ -n "$ENTITY_PIDS" ]; then
  echo "检测到已有 Entity Cluster 进程，kill: $ENTITY_PIDS"
  kill -9 $ENTITY_PIDS
else
  echo "未检测到已有 Entity Cluster 进程"
fi
nohup streamlit run src/streamlit_entity_cluster.py > "$LOG_DIR/streamlit_entity_cluster.log" 2>&1 &

echo
 echo ">>> 全部服务启动完毕！日志目录：$LOG_DIR"
