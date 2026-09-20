#!/usr/bin/env bash
set -eo pipefail
cd /home/jovyan/privado/llm-xray-project

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [QUEUE QWEN36] Iniciando Job 1: Conclusao do Exp 1 Zero-Shot (casos restantes em serie)..."
llm-xray-evaluation/tools/py llm-xray-evaluation/tools/run_zeroshot.py \
  --model qwen3.6-27b \
  --manifest llm-xray-evaluation/inputs/benchmarks/exp1_image_level/manifest.json

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [QUEUE QWEN36] Job 1 concluido! Iniciando Job 2: Exp 1 Multi-Agent Council (45 casos em serie)..."
llm-xray-evaluation/tools/py llm-xray-evaluation/tools/run_opencode.py \
  --mode agents \
  --model qwen3.6-27b \
  --manifest llm-xray-evaluation/inputs/benchmarks/exp1_image_level/manifest.json

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [QUEUE QWEN36] Job 2 concluido! Iniciando Job 3: Exp 2 Multi-Agent Council (48 casos em serie)..."
llm-xray-evaluation/tools/py llm-xray-evaluation/tools/run_opencode.py \
  --mode agents \
  --model qwen3.6-27b \
  --manifest llm-xray-evaluation/inputs/benchmarks/exp2_bbox_localization/manifest.json

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [QUEUE QWEN36] Todos os experimentos da fila do Qwen 3.6 foram concluidos!"
