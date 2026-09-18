"""Baixa e trava (sha256 + revisão) todos os pesos de modelos oferecidos aos agentes.

Uso:
    uv run python tools/setup_models.py            # baixa o que falta e grava/atualiza o lock
    uv run python tools/setup_models.py --verify   # só confere se os pesos em cache batem com o lock

Três origens:
  * torchxrayvision: cache padrão da biblioteca (~/.torchxrayvision/models_data), porque alguns
    baseline models ignoram `cache_dir` e sempre usam esse caminho;
  * Hugging Face: snapshots com revisão fixada em ~/.cache/xray-bench/hf/<repo>, carregados depois
    por caminho local, sem rede;
  * arquivos avulsos (release do GitHub) em ~/.cache/xray-bench/files.

O `models.lock.json` versionado garante que todas as rodadas usem exatamente os mesmos arquivos.
Modelos do torchxrayvision que predizem raça, sexo e idade ficam de fora de propósito: não têm
relação com a tarefa e não devem ser oferecidos aos agentes.
"""

import argparse
import hashlib
import json
import pathlib
import sys
import urllib.request

BENCH = pathlib.Path(__file__).resolve().parent.parent
LOCK_PATH = BENCH / "models.lock.json"
HOME = pathlib.Path.home()
TXV_CACHE = HOME / ".torchxrayvision" / "models_data"
BENCH_CACHE = HOME / ".cache" / "xray-bench"

# nome lógico -> construtor que dispara o download no cache do torchxrayvision
TXV_MODELS = {
    "densenet121-res224-all": lambda xrv: xrv.models.DenseNet(weights="densenet121-res224-all"),
    "densenet121-res224-nih": lambda xrv: xrv.models.DenseNet(weights="densenet121-res224-nih"),
    "densenet121-res224-pc": lambda xrv: xrv.models.DenseNet(weights="densenet121-res224-pc"),
    "densenet121-res224-chex": lambda xrv: xrv.models.DenseNet(weights="densenet121-res224-chex"),
    "densenet121-res224-rsna": lambda xrv: xrv.models.DenseNet(weights="densenet121-res224-rsna"),
    "densenet121-res224-mimic_nb": lambda xrv: xrv.models.DenseNet(weights="densenet121-res224-mimic_nb"),
    "densenet121-res224-mimic_ch": lambda xrv: xrv.models.DenseNet(weights="densenet121-res224-mimic_ch"),
    "resnet50-res512-all": lambda xrv: xrv.models.ResNet(weights="resnet50-res512-all"),
    "resnet-ae-101-elastic": lambda xrv: xrv.autoencoders.ResNetAE(weights="101-elastic"),
    "chestx_det-pspnet": lambda xrv: xrv.baseline_models.chestx_det.PSPNet(),
    "chestx_anatomy-unet-resnet50": lambda xrv: xrv.baseline_models.chestx_anatomy.UNetResNet50(),
    "xinario-view": lambda xrv: xrv.baseline_models.xinario.ViewModel(),
    "jfhealthcare-densenet121": lambda xrv: xrv.baseline_models.jfhealthcare.DenseNet(),
}

# repositório -> (revisão fixada, arquivos necessários)
HF_MODELS = {
    "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224": (
        "9f341de24bfb00180f1b847274256e9b65a3a32e",
        ["open_clip_config.json", "open_clip_pytorch_model.bin", "LICENSE.md", "README.md"],
    ),
    "wanglab/medsam-vit-base": (
        "de8488bca37bb1d4fb190f612c516126d739ce3b",
        ["config.json", "preprocessor_config.json", "pytorch_model.bin", "README.md"],
    ),
    "prithivMLmods/Bone-Fracture-Detection": (
        "7ccaf775b87236a613265389e05c3f360f1215db",
        ["config.json", "preprocessor_config.json", "model.safetensors", "README.md"],
    ),
}

# repositórios carregados por nome (o BiomedCLIP monta a torre de texto e o tokenizer a partir deste),
# baixados no cache padrão do Hugging Face apontado por HF_HUB_CACHE; o lock registra o commit resolvido
HF_BY_NAME = {
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract": ["config.json", "vocab.txt", "tokenizer_config.json"],
}

FILES = {
    "grazpedwri-yolov8-best.pt": "https://github.com/RuiyangJu/Bone_Fracture_Detection_YOLOv8/releases/download/Trained_model/best.pt",
}


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def entry(path: pathlib.Path, **extra) -> dict:
    return {"path": "~/" + str(path.relative_to(HOME)), "sha256": sha256(path), "bytes": path.stat().st_size, **extra}


def fetch_txv() -> dict:
    import torchxrayvision as xrv

    out = {}
    for name, build in TXV_MODELS.items():
        print(f"-> {name}", flush=True)
        model = build(xrv)
        local = getattr(model, "weights_filename_local", None)
        if not local:
            # o autoencoder não guarda o caminho no objeto; o nome do arquivo vem do URL
            local = TXV_CACHE / pathlib.Path(xrv.autoencoders.model_urls[model.weights]["weights_url"]).name
        out[f"txv/{name}"] = entry(pathlib.Path(local))
    return out


def fetch_hf() -> dict:
    from huggingface_hub import hf_hub_download

    out = {}
    for repo, (revision, files) in HF_MODELS.items():
        print(f"-> {repo}@{revision[:10]}", flush=True)
        target = BENCH_CACHE / "hf" / repo
        # o classificador da comunidade só publica os pesos finais em subpastas de checkpoint
        remap = {"model.safetensors": "checkpoint-1108/model.safetensors",
                 "config.json": "checkpoint-1108/config.json",
                 "preprocessor_config.json": "checkpoint-1108/preprocessor_config.json"}
        for name in files:
            remote = remap.get(name, name) if repo.startswith("prithivMLmods/") else name
            downloaded = pathlib.Path(hf_hub_download(repo, remote, revision=revision, cache_dir=BENCH_CACHE / "hub"))
            dest = target / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                dest.write_bytes(downloaded.read_bytes())
            out[f"hf/{repo}/{name}"] = entry(dest, revision=revision)

    for repo, files in HF_BY_NAME.items():
        print(f"-> {repo}", flush=True)
        for name in files:
            downloaded = pathlib.Path(hf_hub_download(repo, name, cache_dir=BENCH_CACHE / "hub"))
            revision = downloaded.parent.name  # .../snapshots/<commit>/<arquivo>
            out[f"hf/{repo}/{name}"] = entry(downloaded, revision=revision)

    # com `local-dir:`, o open_clip procura o tokenizer dentro da própria pasta do BiomedCLIP
    biomedclip = BENCH_CACHE / "hf" / "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    for name in ["config.json", "vocab.txt", "tokenizer_config.json"]:
        source = pathlib.Path(hf_hub_download("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract", name, cache_dir=BENCH_CACHE / "hub"))
        dest = biomedclip / name
        if not dest.exists():
            dest.write_bytes(source.read_bytes())
        out[f"hf/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/{name}"] = entry(dest, revision=source.parent.name)
    return out


def fetch_files() -> dict:
    out = {}
    for name, url in FILES.items():
        print(f"-> {name}", flush=True)
        dest = BENCH_CACHE / "files" / name
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, dest)
        out[f"file/{name}"] = entry(dest, url=url)
    return out


def verify(lock: dict) -> int:
    bad = 0
    for name, item in lock["weights"].items():
        path = pathlib.Path(item["path"]).expanduser()
        if not path.exists():
            print(f"FALTANDO  {name}: {path}")
            bad += 1
        elif sha256(path) != item["sha256"]:
            print(f"DIVERGE   {name}: {path}")
            bad += 1
        else:
            print(f"ok        {name}")
    return 1 if bad else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="não baixa nada, só compara com o lock")
    args = parser.parse_args()

    lock = json.loads(LOCK_PATH.read_text()) if LOCK_PATH.exists() else {}
    if args.verify:
        if not lock:
            print(f"lock inexistente: {LOCK_PATH}", file=sys.stderr)
            return 1
        return verify(lock)

    weights = {**fetch_txv(), **fetch_hf(), **fetch_files()}
    for name, item in weights.items():
        previous = lock.get("weights", {}).get(name)
        if previous and previous["sha256"] != item["sha256"]:
            print(f"ATENÇÃO: {name} mudou em relação ao lock")

    import torchxrayvision as xrv

    LOCK_PATH.write_text(
        json.dumps({"torchxrayvision": xrv.__version__, "weights": weights}, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"lock gravado em {LOCK_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
