#!/usr/bin/env bash
set -eo pipefail
cd /home/jovyan/privado/llm-xray-project

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Gemma4-W1] Iniciando Stage 1: Exp 2 Single-Agent (cases 0..24)..."
llm-xray-evaluation/tools/py llm-xray-evaluation/tools/run_opencode.py \
  --mode single \
  --model gemma4-26b \
  --manifest llm-xray-evaluation/inputs/benchmarks/exp2_bbox_localization/manifest.json \
  --case-range 0 24

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Gemma4-W1] Stage 1 concluido! Iniciando Stage 2: Exp 2 Multi-Agent Council (cases 0..24)..."
llm-xray-evaluation/tools/py llm-xray-evaluation/tools/run_opencode.py \
  --mode agents \
  --model gemma4-26b \
  --manifest llm-xray-evaluation/inputs/benchmarks/exp2_bbox_localization/manifest.json \
  --case-range 0 24

echo "[$(date '+%Y-%m-%d %H:%M:%S')] [Gemma4-W1] Todos os estagios concluidos com sucesso!"
