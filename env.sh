# Confina tudo o que roda neste projeto dentro de llm-xray-evaluation/.
# Uso: source llm-xray-evaluation/env.sh   (antes de qualquer uv, python ou opencode)
#
# Nada pode ser escrito fora da pasta do projeto: caches, temporários, pesos de modelos, sessões do
# opencode e configurações de bibliotecas são redirecionados para .sandbox/ e models/.

EVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EVAL
SANDBOX="$EVAL/.sandbox"
export SANDBOX

# arquivos temporários
export TMPDIR="$SANDBOX/tmp"

# uv: cache de pacotes e intérpretes gerenciados
export UV_CACHE_DIR="$SANDBOX/uv-cache"
export UV_PYTHON_INSTALL_DIR="$SANDBOX/uv-python"
export UV_TOOL_DIR="$SANDBOX/uv-tools"
export UV_PROJECT_ENVIRONMENT="$EVAL/.venv"

# bibliotecas Python que gravam em ~/.cache, ~/.config etc.
export XDG_CACHE_HOME="$SANDBOX/home/.cache"
export MPLCONFIGDIR="$SANDBOX/home/.config/matplotlib"
export TORCH_HOME="$EVAL/models/torch"
export HF_HOME="$SANDBOX/home/.cache/huggingface"
export HF_HUB_CACHE="$EVAL/models/hf-hub"
export HF_HUB_DISABLE_TELEMETRY=1
export YOLO_CONFIG_DIR="$SANDBOX/home/.config/Ultralytics"
export XRAY_BENCH_MODELS="$EVAL/models"

# opencode: configuração, sessões, cache e estado ficam no sandbox
export XDG_CONFIG_HOME="$SANDBOX/opencode/config"
export XDG_DATA_HOME="$SANDBOX/opencode/data"
export XDG_STATE_HOME="$SANDBOX/opencode/state"
export OPENCODE_DISABLE_AUTOUPDATE=1

# credenciais do endpoint da DGX (arquivo fora do git)
if [ -f "$EVAL/.env" ]; then
    set -a
    . "$EVAL/.env"
    set +a
fi

mkdir -p "$TMPDIR" "$UV_CACHE_DIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$YOLO_CONFIG_DIR" "$HF_HOME" \
    "$HF_HUB_CACHE" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$XDG_STATE_HOME" "$SANDBOX/home"

# o torchxrayvision (e alguns baseline models) grava sempre em ~/.torchxrayvision; o HOME dos processos
# Python aponta para $SANDBOX/home (ver tools/py), onde este link leva aos pesos em models/
[ -e "$SANDBOX/home/.torchxrayvision" ] || ln -s "$EVAL/models/torchxrayvision" "$SANDBOX/home/.torchxrayvision"

# configuração do opencode versionada no projeto (provedor DGX-UFSC; a chave vem do .env)
export OPENCODE_CONFIG="$EVAL/harness/opencode/opencode.json"
