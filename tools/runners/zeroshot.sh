#!/usr/bin/env bash
# Grupo sem ferramentas de um modelo: zero-shot aberto e fechado (Exp 1 e 2) e zero-shot com os escores dos
# classificadores (Exp 1), com orcamento de pensamento de 6.000 tokens. O queue.sh roda os mesmos passos no
# comeco da fila; este script serve para refazer so esse grupo. Casos ja respondidos sao pulados.
# Uso: setsid nohup bash llm-xray-evaluation/tools/runners/zeroshot.sh <modelo> >> llm-xray-evaluation/zeroshot_<modelo>.log 2>&1 < /dev/null &
MODEL="$1"
[ -n "$MODEL" ] || { echo "uso: zeroshot.sh <modelo>"; exit 2; }
cd /home/jovyan/privado/llm-xray-project
source llm-xray-evaluation/env.sh
PY=llm-xray-evaluation/tools/py
EXP1=llm-xray-evaluation/inputs/benchmarks/exp1_image_level/manifest.json
EXP2=llm-xray-evaluation/inputs/benchmarks/exp2_bbox_localization/manifest.json
step() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$MODEL] $*"; }
for version in v4 v4c; do
    step "zero-shot $version"
    $PY llm-xray-evaluation/tools/run_zeroshot.py --version $version --model "$MODEL" --manifest $EXP1
    $PY llm-xray-evaluation/tools/run_zeroshot_exp2.py --version $version --model "$MODEL" --manifest $EXP2
    for clf in densenet121-res224-all densenet121-res224-pc; do
        step "zero-shot com escores $clf $version"
        $PY llm-xray-evaluation/tools/run_zeroshot.py --version $version --classifier $clf --model "$MODEL" --manifest $EXP1
    done
done
step "grupo sem ferramentas concluido"
