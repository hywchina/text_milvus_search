#!/usr/bin/env bash
set -euo pipefail

# 在每行输出前加上时间戳的函数
log() {
  local ts
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[$ts] $*"
}

# 日志目录
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

log "===== 1. 启动/重启 Milvus (Docker standalone) ====="
# stop 会在未运行时返回非零，忽略错误继续
bash milvus_standalone/standalone_embed.sh stop 2>/dev/null || true
bash milvus_standalone/standalone_embed.sh start

echo
log "===== 2. 启动 Embedding 服务 (port 8001) ====="
EMB_PORT=8001
EMB_PIDS=$(lsof -t -i tcp:$EMB_PORT || true)
if [ -n "$EMB_PIDS" ]; then
  log "Port $EMB_PORT 被占用，kill 进程: $EMB_PIDS"
  kill -9 $EMB_PIDS
else
  log "Port $EMB_PORT 未被占用"
fi
# 后台启动，并将带时间戳的日志写入
nohup bash -c "python src/api_embedding.py 2>&1 | while IFS= read -r line; do printf '[%s] %s\n' \"\$(date '+%Y-%m-%d %H:%M:%S')\" \"\$line\"; done" >> "$LOG_DIR/api_embedding.log" &

echo
log "===== 3. 启动 Milvus API 服务 (port 8002) ====="
API_PORT=8002
API_PIDS=$(lsof -t -i tcp:$API_PORT || true)
if [ -n "$API_PIDS" ]; then
  log "Port $API_PORT 被占用，kill 进程: $API_PIDS"
  kill -9 $API_PIDS
else
  log "Port $API_PORT 未被占用"
fi
nohup bash -c "python src/api_search_milvus.py 2>&1 | while IFS= read -r line; do printf '[%s] %s\n' \"\$(date '+%Y-%m-%d %H:%M:%S')\" \"\$line\"; done" >> "$LOG_DIR/api_search_milvus.log" &

echo
log "===== 4. 启动 Streamlit 界面 ====="
STREAMLIT_PIDS=$(pgrep -f "streamlit_milvus_search.py" || true)
if [ -n "$STREAMLIT_PIDS" ]; then
  log "检测到已有 Streamlit 进程，kill: $STREAMLIT_PIDS"
  kill -9 $STREAMLIT_PIDS
else
  log "未检测到已有 Streamlit 进程"
fi
nohup bash -c "streamlit run src/streamlit_milvus_search.py 2>&1 | while IFS= read -r line; do printf '[%s] %s\n' \"\$(date '+%Y-%m-%d %H:%M:%S')\" \"\$line\"; done" >> "$LOG_DIR/streamlit_milvus_search.log" &

echo
log "===== 5. 启动 Entity Cluster 界面 ====="
ENTITY_PIDS=$(pgrep -f "streamlit_entity_cluster.py" || true)
if [ -n "$ENTITY_PIDS" ]; then
  log "检测到已有 Entity Cluster 进程，kill: $ENTITY_PIDS"
  kill -9 $ENTITY_PIDS
else
  log "未检测到已有 Entity Cluster 进程"
fi
nohup bash -c "streamlit run src/streamlit_entity_cluster.py 2>&1 | while IFS= read -r line; do printf '[%s] %s\n' \"\$(date '+%Y-%m-%d %H:%M:%S')\" \"\$line\"; done" >> "$LOG_DIR/streamlit_entity_cluster.log" &

echo
log "===== 6. 启动 Milvus Text Search & Recommend 界面 ====="
ENTITY_PIDS=$(pgrep -f "streamlit_milvus_search_recommend.py" || true)
if [ -n "$ENTITY_PIDS" ]; then
  log "检测到已有 streamlit_milvus_search_recommend 进程，kill: $ENTITY_PIDS"
  kill -9 $ENTITY_PIDS
else
  log "未检测到已有 streamlit_milvus_search_recommend 进程"
fi
nohup bash -c "streamlit run src/streamlit_milvus_search_recommend.py 2>&1 | while IFS= read -r line; do printf '[%s] %s\n' \"\$(date '+%Y-%m-%d %H:%M:%S')\" \"\$line\"; done" >> "$LOG_DIR/streamlit_milvus_search_recommend.log" &

echo
log ">>> 全部服务启动完毕！日志目录：$LOG_DIR"
