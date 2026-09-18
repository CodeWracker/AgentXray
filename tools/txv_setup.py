"""Baixa e trava (sha256) os pesos do torchxrayvision usados na bancada.

Uso:
    uv run python tools/txv_setup.py            # baixa o que falta e grava/atualiza o lock
    uv run python tools/txv_setup.py --verify   # só confere se os pesos em cache batem com o lock

Os pesos ficam no cache padrão da biblioteca (~/.torchxrayvision/models_data), porque alguns
baseline models ignoram `cache_dir` e sempre usam esse caminho. O arquivo `weights.lock.json`
versionado garante que todas as rodadas usem exatamente os mesmos arquivos.

Os modelos que predizem raça, sexo e idade ficam de fora de propósito: não têm relação com a
tarefa e não devem ser oferecidos aos agentes.
"""

import argparse
import hashlib
import json
import pathlib
import sys

import torchxrayvision as xrv

LOCK_PATH = pathlib.Path(__file__).resolve().parent.parent / "weights.lock.json"
CACHE_DIR = pathlib.Path(xrv.utils.get_cache_dir())

# nome lógico -> construtor que dispara o download
MODELS = {
    "densenet121-res224-all": lambda: xrv.models.DenseNet(weights="densenet121-res224-all"),
    "densenet121-res224-nih": lambda: xrv.models.DenseNet(weights="densenet121-res224-nih"),
    "densenet121-res224-pc": lambda: xrv.models.DenseNet(weights="densenet121-res224-pc"),
    "densenet121-res224-chex": lambda: xrv.models.DenseNet(weights="densenet121-res224-chex"),
    "densenet121-res224-rsna": lambda: xrv.models.DenseNet(weights="densenet121-res224-rsna"),
    "densenet121-res224-mimic_nb": lambda: xrv.models.DenseNet(weights="densenet121-res224-mimic_nb"),
    "densenet121-res224-mimic_ch": lambda: xrv.models.DenseNet(weights="densenet121-res224-mimic_ch"),
    "resnet50-res512-all": lambda: xrv.models.ResNet(weights="resnet50-res512-all"),
    "resnet-ae-101-elastic": lambda: xrv.autoencoders.ResNetAE(weights="101-elastic"),
    "chestx_det-pspnet": lambda: xrv.baseline_models.chestx_det.PSPNet(),
    "chestx_anatomy-unet-resnet50": lambda: xrv.baseline_models.chestx_anatomy.UNetResNet50(),
    "xinario-view": lambda: xrv.baseline_models.xinario.ViewModel(),
    "jfhealthcare-densenet121": lambda: xrv.baseline_models.jfhealthcare.DenseNet(),
}


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def weights_file(model) -> pathlib.Path:
    local = getattr(model, "weights_filename_local", None)
    if local:
        return pathlib.Path(local)
    # o autoencoder não guarda o caminho no objeto; o nome do arquivo vem do URL
    url = xrv.autoencoders.model_urls[model.weights]["weights_url"]
    return CACHE_DIR / pathlib.Path(url).name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="não baixa nada, só compara com o lock")
    args = parser.parse_args()

    lock = json.loads(LOCK_PATH.read_text()) if LOCK_PATH.exists() else {}

    if args.verify:
        if not lock:
            print(f"lock inexistente: {LOCK_PATH}", file=sys.stderr)
            return 1
        bad = 0
        for name, entry in lock["weights"].items():
            path = CACHE_DIR / entry["file"]
            if not path.exists():
                print(f"FALTANDO  {name}: {path}")
                bad += 1
            elif sha256(path) != entry["sha256"]:
                print(f"DIVERGE   {name}: {path}")
                bad += 1
            else:
                print(f"ok        {name}")
        return 1 if bad else 0

    weights = {}
    for name, build in MODELS.items():
        print(f"-> {name}", flush=True)
        model = build()
        path = weights_file(model)
        digest = sha256(path)
        previous = lock.get("weights", {}).get(name)
        if previous and previous["sha256"] != digest:
            print(f"ATENÇÃO: {name} mudou em relação ao lock ({previous['sha256'][:12]} -> {digest[:12]})")
        weights[name] = {"file": path.name, "sha256": digest, "bytes": path.stat().st_size}

    LOCK_PATH.write_text(
        json.dumps({"torchxrayvision": xrv.__version__, "weights": weights}, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"lock gravado em {LOCK_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
