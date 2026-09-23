#!/usr/bin/env bash
# Encadeia a fila principal e a fila da escada de um modelo, depois de esperar qualquer rodada do opencode desse
# modelo que ainda esteja no ar (para nunca haver dois casos no mesmo modelo).
# Uso: setsid nohup bash llm-xray-evaluation/tools/runners/queue_chain.sh <modelo> > /dev/null 2>&1 < /dev/null &
MODEL="$1"
[ -n "$MODEL" ] || { echo "uso: queue_chain.sh <modelo>"; exit 2; }
cd /home/jovyan/privado/llm-xray-project
while pgrep -f "tools/run_opencode.py .*--model $MODEL " >/dev/null; do
    sleep 300
done
bash llm-xray-evaluation/tools/runners/queue.sh "$MODEL" v4 >> "llm-xray-evaluation/queue_v4_$MODEL.log" 2>&1
bash llm-xray-evaluation/tools/runners/queue_ladder.sh "$MODEL" >> "llm-xray-evaluation/queue_ladder_$MODEL.log" 2>&1
