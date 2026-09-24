#!/usr/bin/env bash
# Fila completa de um modelo: condicao aberta (v4), degraus da escada e condicao fechada (v4c), em serie.
# Dentro de cada condicao do opencode, varios casos rodam ao mesmo tempo no mesmo motor: WORKERS_SINGLE para o
# agente unico e WORKERS_COUNCIL para o conselho (cada conselho ja faz varias requisicoes simultaneas). O numero
# fica registrado em cada harness_result.json. Os valores padrao sao os mesmos para os tres modelos, para o tempo
# de parede de uma condicao ser comparavel entre eles.
# Uso: setsid nohup bash llm-xray-evaluation/tools/runners/queue.sh <modelo> >> llm-xray-evaluation/queue_<modelo>.log 2>&1 < /dev/null &
# Sem "set -e": uma condicao que falha nao impede as seguintes; o que ja foi concluido e pulado ao religar.
set -o pipefail
MODEL="$1"
[ -n "$MODEL" ] || { echo "uso: queue.sh <modelo>"; exit 2; }
cd /home/jovyan/privado/llm-xray-project
source llm-xray-evaluation/env.sh

PY=llm-xray-evaluation/tools/py
EXP1=llm-xray-evaluation/inputs/benchmarks/exp1_image_level/manifest.json
EXP2=llm-xray-evaluation/inputs/benchmarks/exp2_bbox_localization/manifest.json
SINGLE_TIMEOUT=240
COUNCIL_TIMEOUT=480
WORKERS_SINGLE="${WORKERS_SINGLE:-6}"
WORKERS_COUNCIL="${WORKERS_COUNCIL:-2}"
# classificadores cujas saidas vao no prompt do degrau modelo + classificador: um que viu o ChestX-ray14 e um que nao
CLASSIFIERS="densenet121-res224-all densenet121-res224-pc"

step() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$MODEL] $*"; }
single() { $PY llm-xray-evaluation/tools/run_opencode.py --version "$1" --mode single --model "$MODEL" --timeout $SINGLE_TIMEOUT --workers $WORKERS_SINGLE --manifest "$2" "${@:3}"; }
council() { $PY llm-xray-evaluation/tools/run_opencode.py --version "$1" --mode agents --model "$MODEL" --timeout $COUNCIL_TIMEOUT --workers $WORKERS_COUNCIL --manifest "$2"; }

step "1/9 zero-shot aberto e fechado, Exp 1 e 2"
for version in v4 v4c; do
    $PY llm-xray-evaluation/tools/run_zeroshot.py --version $version --model "$MODEL" --manifest $EXP1
    $PY llm-xray-evaluation/tools/run_zeroshot_exp2.py --version $version --model "$MODEL" --manifest $EXP2
done
step "2/9 modelo + classificador no prompt, aberto e fechado (Exp 1)"
for clf in $CLASSIFIERS; do
    for version in v4 v4c; do
        $PY llm-xray-evaluation/tools/run_zeroshot.py --version $version --classifier $clf --model "$MODEL" --manifest $EXP1
    done
done
step "3/9 agente unico aberto Exp 1";  single v4 $EXP1
step "4/9 agente unico aberto Exp 2";  single v4 $EXP2
step "5/9 conselho aberto Exp 1";      council v4 $EXP1
step "6/9 conselho aberto Exp 2";      council v4 $EXP2
step "7/9 agente unico fechado Exp 1 e 2"
single v4c $EXP1
single v4c $EXP2
step "8/9 repeticoes r2 e r3 do agente unico aberto Exp 1 (consistencia)"
single v4 $EXP1 --repeat 3
step "9/9 conselho fechado Exp 1 e 2"
council v4c $EXP1
council v4c $EXP2
step "complemento: zero-shot sem saida (refaz so os casos faltantes)"
for version in v4 v4c; do
    $PY llm-xray-evaluation/tools/run_zeroshot.py --version $version --model "$MODEL" --manifest $EXP1
    $PY llm-xray-evaluation/tools/run_zeroshot_exp2.py --version $version --model "$MODEL" --manifest $EXP2
done
step "fila concluida"
