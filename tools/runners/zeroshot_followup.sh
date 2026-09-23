#!/usr/bin/env bash
# Espera a fila v3 do modelo terminar e refaz so os casos zero-shot sem saida (os existentes sao pulados).
# Existe porque o zero-shot do Exp 1 nao lia respostas que vinham so no campo de raciocinio; roda depois
# da fila para nunca haver duas requisicoes simultaneas no mesmo modelo.
# Uso: nohup bash llm-xray-evaluation/tools/runners/zeroshot_followup.sh <modelo> <pid da fila> >> <log> 2>&1 &
MODEL="$1"; QUEUE_PID="$2"
cd /home/jovyan/privado/llm-xray-project
source llm-xray-evaluation/env.sh
while kill -0 "$QUEUE_PID" 2>/dev/null; do sleep 600; done
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [V3 $MODEL] complemento zero-shot"
llm-xray-evaluation/tools/py llm-xray-evaluation/tools/run_zeroshot.py --version v3 --model "$MODEL" --manifest llm-xray-evaluation/inputs/benchmarks/exp1_image_level/manifest.json
llm-xray-evaluation/tools/py llm-xray-evaluation/tools/run_zeroshot_exp2.py --version v3 --model "$MODEL" --manifest llm-xray-evaluation/inputs/benchmarks/exp2_bbox_localization/manifest.json
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [V3 $MODEL] complemento zero-shot concluido"
