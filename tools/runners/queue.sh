#!/usr/bin/env bash
# Fila de um modelo: todas as condicoes em serie; dentro de cada condicao do opencode, WORKERS casos ao mesmo tempo (padrao 1).
# Uso: WORKERS=4 nohup bash llm-xray-evaluation/tools/runners/queue.sh <modelo> [versao] >> llm-xray-evaluation/queue_<versao>_<modelo>.log 2>&1 &
# Sem "set -e": uma condicao que falha nao impede as seguintes; as rodadas ja concluidas sao puladas ao religar.
set -o pipefail
MODEL="$1"
VERSION="${2:-v4}"
[ -n "$MODEL" ] || { echo "uso: queue.sh <modelo> [versao]"; exit 2; }
cd /home/jovyan/privado/llm-xray-project
source llm-xray-evaluation/env.sh

PY=llm-xray-evaluation/tools/py
EXP1=llm-xray-evaluation/inputs/benchmarks/exp1_image_level/manifest.json
EXP2=llm-xray-evaluation/inputs/benchmarks/exp2_bbox_localization/manifest.json
SINGLE_TIMEOUT=240
COUNCIL_TIMEOUT=480
WORKERS="${WORKERS:-1}"

step() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$VERSION $MODEL] $*"; }

step "1/7 zero-shot Exp 1"
$PY llm-xray-evaluation/tools/run_zeroshot.py --version $VERSION --model "$MODEL" --manifest $EXP1
step "2/7 zero-shot Exp 2"
$PY llm-xray-evaluation/tools/run_zeroshot_exp2.py --version $VERSION --model "$MODEL" --manifest $EXP2
step "3/7 single-agent Exp 1"
$PY llm-xray-evaluation/tools/run_opencode.py --version $VERSION --mode single --model "$MODEL" --timeout $SINGLE_TIMEOUT --workers $WORKERS --manifest $EXP1
step "4/7 single-agent Exp 2"
$PY llm-xray-evaluation/tools/run_opencode.py --version $VERSION --mode single --model "$MODEL" --timeout $SINGLE_TIMEOUT --workers $WORKERS --manifest $EXP2
step "5/7 conselho Exp 1"
$PY llm-xray-evaluation/tools/run_opencode.py --version $VERSION --mode agents --model "$MODEL" --timeout $COUNCIL_TIMEOUT --workers $WORKERS --manifest $EXP1
step "6/7 conselho Exp 2"
$PY llm-xray-evaluation/tools/run_opencode.py --version $VERSION --mode agents --model "$MODEL" --timeout $COUNCIL_TIMEOUT --workers $WORKERS --manifest $EXP2
step "7/7 repeticoes r2 e r3 do single-agent Exp 1 (consistencia)"
$PY llm-xray-evaluation/tools/run_opencode.py --version $VERSION --mode single --model "$MODEL" --timeout $SINGLE_TIMEOUT --workers $WORKERS --manifest $EXP1 --repeat 3
step "complemento: zero-shot sem saida (refaz so os casos faltantes)"
$PY llm-xray-evaluation/tools/run_zeroshot.py --version $VERSION --model "$MODEL" --manifest $EXP1
$PY llm-xray-evaluation/tools/run_zeroshot_exp2.py --version $VERSION --model "$MODEL" --manifest $EXP2
step "fila concluida"
