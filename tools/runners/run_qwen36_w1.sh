#!/usr/bin/env bash
set -eo pipefail
cd /home/jovyan/privado/llm-xray-project

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Qwen3.6-W1] Iniciando Stage 1: Exp 1 Zero-Shot (45 casos)..."
llm-xray-evaluation/tools/py llm-xray-evaluation/tools/run_zeroshot.py \
  --model qwen3.6-27b \
  --manifest llm-xray-evaluation/inputs/benchmarks/exp1_image_level/manifest.json

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Qwen3.6-W1] Stage 1 concluido! Iniciando Stage 2: Exp 1 Multi-Agent Council (cases 0..23)..."
llm-xray-evaluation/tools/py llm-xray-evaluation/tools/run_opencode.py \
  --mode agents \
  --model qwen3.6-27b \
  --manifest llm-xray-evaluation/inputs/benchmarks/exp1_image_level/manifest.json \
  --case-range 0 23

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Qwen3.6-W1] Stage 2 concluido! Iniciando Stage 3: Exp 2 Multi-Agent Council (cases 0..24)..."
llm-xray-evaluation/tools/py llm-xray-evaluation/tools/run_opencode.py \
  --mode agents \
  --model qwen3.6-27b \
  --manifest llm-xray-evaluation/inputs/benchmarks/exp2_bbox_localization/manifest.json \
  --case-range 0 24

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Qwen3.6-W1] Todos os estagios concluidos com sucesso!"
