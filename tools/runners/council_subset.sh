#!/usr/bin/env bash
# Conselho aberto so no subconjunto fixo de casos (tools/sample_council_subset.py), para modelos em que o conselho completo
# e inviavel (emenda post hoc de 30/09/2026: qwen3.8-27b). As rodadas vao para as mesmas pastas do conselho aberto.
# Uso: setsid nohup bash llm-xray-evaluation/tools/runners/council_subset.sh <modelo> >> llm-xray-evaluation/queue_<modelo>.log 2>&1 < /dev/null &
set -o pipefail
MODEL="$1"
[ -n "$MODEL" ] || { echo "uso: council_subset.sh <modelo>"; exit 2; }
cd /home/jovyan/privado/llm-xray-project
source llm-xray-evaluation/env.sh
PY=llm-xray-evaluation/tools/py
WORKERS_COUNCIL="${WORKERS_COUNCIL:-1}"
for e in exp1_image_level exp2_bbox_localization; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$MODEL] conselho aberto, subconjunto de $e"
    $PY llm-xray-evaluation/tools/run_opencode.py --version v4 --mode agents --model "$MODEL" --timeout 480 \
        --workers "$WORKERS_COUNCIL" --manifest "llm-xray-evaluation/inputs/benchmarks/$e/manifest_council_subset.json"
done
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$MODEL] subconjunto de conselhos concluido"
