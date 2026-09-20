#!/usr/bin/env bash
set -eo pipefail
cd /home/jovyan/privado/llm-xray-project

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Qwen3.8-W2] Iniciando Stage 1: Exp 1 Multi-Agent Council (cases 23..45)..."
llm-xray-evaluation/tools/py llm-xray-evaluation/tools/run_opencode.py \
  --mode agents \
  --model qwen3.8-27b \
  --manifest llm-xray-evaluation/inputs/benchmarks/exp1_image_level/manifest.json \
  --case-range 23 45

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Qwen3.8-W2] Stage 1 concluido! Iniciando Stage 2: Exp 2 Multi-Agent Council (cases 24..48)..."
llm-xray-evaluation/tools/py llm-xray-evaluation/tools/run_opencode.py \
  --mode agents \
  --model qwen3.8-27b \
  --manifest llm-xray-evaluation/inputs/benchmarks/exp2_bbox_localization/manifest.json \
  --case-range 24 48

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Qwen3.8-W2] Todos os estagios concluidos com sucesso!"
