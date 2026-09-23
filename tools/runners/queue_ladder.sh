#!/usr/bin/env bash
# Fila da escada de comparacao de um modelo. Espera a fila principal do mesmo modelo (queue.sh) terminar, para nunca
# haver dois casos no mesmo modelo, e roda em serie, um caso por vez:
#   1. zero-shot aberto de novo, agora gravando tokens e tempo (o anterior vai para results/v4-zeroshot-sem-tokens/)
#   2. zero-shot com vocabulario fechado (v4c)
#   3. modelo + saida do classificador no prompt, sem ferramentas, aberto e fechado, com um classificador que viu o
#      ChestX-ray14 (densenet121-res224-all) e um que nunca viu (densenet121-res224-pc)
#   4. agente unico com vocabulario fechado, Exp 1 e Exp 2
#   5. conselho com vocabulario fechado, Exp 1 e Exp 2
# Uso: nohup bash llm-xray-evaluation/tools/runners/queue_ladder.sh <modelo> >> llm-xray-evaluation/queue_ladder_<modelo>.log 2>&1 &
# Sem "set -e": uma condicao que falha nao impede as seguintes; o que ja foi concluido e pulado ao religar.
set -o pipefail
MODEL="$1"
[ -n "$MODEL" ] || { echo "uso: queue_ladder.sh <modelo>"; exit 2; }
cd /home/jovyan/privado/llm-xray-project
source llm-xray-evaluation/env.sh

PY=llm-xray-evaluation/tools/py
RES=llm-xray-evaluation/results
EXP1=llm-xray-evaluation/inputs/benchmarks/exp1_image_level/manifest.json
EXP2=llm-xray-evaluation/inputs/benchmarks/exp2_bbox_localization/manifest.json
SINGLE_TIMEOUT=240
COUNCIL_TIMEOUT=480
CLASSIFIERS="densenet121-res224-all densenet121-res224-pc"

step() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ladder $MODEL] $*"; }

while pgrep -f "runners/queue.sh $MODEL " >/dev/null; do
    sleep 300
done
step "fila principal terminou; comecando a escada"

step "1/5 zero-shot aberto com tokens"
for bench in exp1_image_level exp2_bbox_localization; do
    dir=$RES/v4/zeroshot/$bench/$MODEL
    if [ -d "$dir" ] && [ ! -d "$dir/_usage" ]; then
        mkdir -p "$RES/v4-zeroshot-sem-tokens/$bench"
        mv "$dir" "$RES/v4-zeroshot-sem-tokens/$bench/$MODEL"
    fi
done
$PY llm-xray-evaluation/tools/run_zeroshot.py --version v4 --model "$MODEL" --manifest $EXP1
$PY llm-xray-evaluation/tools/run_zeroshot_exp2.py --version v4 --model "$MODEL" --manifest $EXP2

step "2/5 zero-shot fechado"
$PY llm-xray-evaluation/tools/run_zeroshot.py --version v4c --model "$MODEL" --manifest $EXP1
$PY llm-xray-evaluation/tools/run_zeroshot_exp2.py --version v4c --model "$MODEL" --manifest $EXP2

step "3/5 modelo + classificador no prompt (Exp 1)"
for clf in $CLASSIFIERS; do
    for version in v4 v4c; do
        $PY llm-xray-evaluation/tools/run_zeroshot.py --version $version --classifier $clf --model "$MODEL" --manifest $EXP1
    done
done

step "4/5 agente unico fechado"
$PY llm-xray-evaluation/tools/run_opencode.py --version v4c --mode single --model "$MODEL" --timeout $SINGLE_TIMEOUT --manifest $EXP1
$PY llm-xray-evaluation/tools/run_opencode.py --version v4c --mode single --model "$MODEL" --timeout $SINGLE_TIMEOUT --manifest $EXP2

step "5/5 conselho fechado"
$PY llm-xray-evaluation/tools/run_opencode.py --version v4c --mode agents --model "$MODEL" --timeout $COUNCIL_TIMEOUT --manifest $EXP1
$PY llm-xray-evaluation/tools/run_opencode.py --version v4c --mode agents --model "$MODEL" --timeout $COUNCIL_TIMEOUT --manifest $EXP2

step "escada concluida"
