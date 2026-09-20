#!/usr/bin/env bash
set -eo pipefail
cd /home/jovyan/privado/llm-xray-project

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [QUEUE GEMMA4] Iniciando Job 1: Exp 2 Single-Agent (48 casos em serie)..."
llm-xray-evaluation/tools/py llm-xray-evaluation/tools/run_opencode.py \
  --mode single \
  --model gemma4-26b \
  --manifest llm-xray-evaluation/inputs/benchmarks/exp2_bbox_localization/manifest.json

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [QUEUE GEMMA4] Job 1 concluido! Iniciando Job 2: Exp 2 Multi-Agent Council (48 casos em serie)..."
llm-xray-evaluation/tools/py llm-xray-evaluation/tools/run_opencode.py \
  --mode agents \
  --model gemma4-26b \
  --manifest llm-xray-evaluation/inputs/benchmarks/exp2_bbox_localization/manifest.json

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [QUEUE GEMMA4] Todos os experimentos da fila do Gemma 4 foram concluidos!"
