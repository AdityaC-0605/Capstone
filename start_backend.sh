#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_ACTIVATE="$ROOT_DIR/venv/bin/activate"
MAIN_LOG="/tmp/mj_main_api.log"
INFER_LOG="/tmp/mj_inference_api.log"
MAIN_PORT="8000"
INFER_PORT="8001"

if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "[ERROR] venv not found at $VENV_ACTIVATE"
  exit 1
fi

cd "$ROOT_DIR"
source "$VENV_ACTIVATE"
export ENVIRONMENT="development"

kill_port() {
  local port="$1"
  local pids
  pids=$(lsof -ti tcp:"$port" || true)
  if [[ -n "$pids" ]]; then
    echo "[INFO] Releasing port $port (PIDs: $pids)"
    kill $pids || true
    sleep 1
  fi
}

kill_port "$MAIN_PORT"
kill_port "$INFER_PORT"

: > "$MAIN_LOG"
: > "$INFER_LOG"

echo "[INFO] Starting main API on :$MAIN_PORT"
nohup uvicorn app.api.main:app --host 0.0.0.0 --port "$MAIN_PORT" --reload > "$MAIN_LOG" 2>&1 &
MAIN_PID=$!

echo "[INFO] Starting inference API on :$INFER_PORT"
nohup python -c "from app.api.inference_service import run_inference_service; run_inference_service(port=$INFER_PORT)" > "$INFER_LOG" 2>&1 &
INFER_PID=$!

sleep 3

API_KEY=$(grep -m1 -o 'sk-test-[^ ]*' "$INFER_LOG" || true)

echo ""
echo "[OK] Started"
echo "  Main API PID:      $MAIN_PID"
echo "  Inference API PID: $INFER_PID"
echo "  Main log:          $MAIN_LOG"
echo "  Inference log:     $INFER_LOG"
if [[ -n "$API_KEY" ]]; then
  echo "  Inference API key: $API_KEY"
else
  echo "  Inference API key: <not found yet, check $INFER_LOG>"
fi

echo ""
echo "[Health checks]"
echo "  curl -s http://localhost:$MAIN_PORT/health"
echo "  curl -s http://localhost:$MAIN_PORT/ready"
echo "  curl -s http://localhost:$MAIN_PORT/api/v1/status"

echo ""
echo "[Stop services]"
echo "  kill $MAIN_PID $INFER_PID"
