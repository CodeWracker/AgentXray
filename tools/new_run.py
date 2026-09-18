"""Prepara um diretório de rodada isolado, com o prompt renderizado e um manifesto de proveniência.

Uso:
    uv run python tools/new_run.py --mode single --model sonnet5
    uv run python tools/new_run.py --mode agents --model qwen3.8-27b --images inputs/ID003-xray.png
    uv run python tools/new_run.py --mode single --model sonnet5 --root /tmp/xray-runs   # fora do repo

Cria `<root>/<versão>/<modo>/<modelo>/<timestamp>/` com cópias das imagens, `PROMPT.md` (o texto exato
a ser entregue ao agente) e `run_manifest.json` (hashes do prompt, das imagens, dos pesos, commit do
repositório e versões das bibliotecas). Depois é só abrir o agente nesse diretório e passar o PROMPT.md.
"""

import argparse
import datetime
import hashlib
import json
import pathlib
import platform
import shutil
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
PROMPTS = REPO / "prompts"
MODES = {"single": "single-agent", "agents": "multi-agent-single-model"}


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def library_versions() -> dict:
    import cv2
    import numpy
    import PIL
    import scipy
    import skimage
    import torch
    import torchvision
    import torchxrayvision

    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "opencv": cv2.__version__,
        "scikit-image": skimage.__version__,
        "pillow": PIL.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torchxrayvision": torchxrayvision.__version__,
    }


def render(template: str, values: dict) -> str:
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", value)
    if "{{" in template:
        raise SystemExit("placeholder sem valor no prompt renderizado")
    return template


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--model", required=True, help="nome do modelo avaliado, ex.: sonnet5, qwen3.8-27b")
    parser.add_argument("--version", default="v2", help="versão dos prompts em prompts/<versão>/")
    parser.add_argument("--images", nargs="+", type=pathlib.Path, default=sorted((REPO / "inputs").glob("*.png")))
    parser.add_argument("--root", type=pathlib.Path, default=REPO / "results", help="onde criar a rodada")
    args = parser.parse_args()

    # confere os pesos antes de gastar uma rodada com um ambiente divergente
    verify = subprocess.run([sys.executable, str(REPO / "tools" / "txv_setup.py"), "--verify"], capture_output=True, text=True)
    if verify.returncode != 0:
        print(verify.stdout, file=sys.stderr)
        raise SystemExit("pesos do torchxrayvision ausentes ou divergentes; rode tools/txv_setup.py")

    import torchxrayvision

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = (args.root / args.version / MODES[args.mode] / args.model / stamp).resolve()
    run_dir.mkdir(parents=True)

    images = []
    for image in args.images:
        shutil.copy2(image, run_dir / image.name)
        images.append({"file": image.name, "sha256": sha256(image)})

    python_cmd = f"uv run --project {REPO} python"
    values = {
        "MODEL_NAME": args.model,
        "IMAGES": ", ".join(f"`{i['file']}`" for i in images),
        "RUN_DIR": str(run_dir),
        "PYTHON": python_cmd,
        "TXV_SOURCE": str(pathlib.Path(torchxrayvision.__file__).parent),
        "TXV_VERSION": torchxrayvision.__version__,
    }
    parts = [PROMPTS / args.version / f"{args.mode}.md", *sorted((PROMPTS / args.version / "_shared").glob("*.md"))]
    prompt = "\n\n".join(render(p.read_text(), values).strip() for p in parts) + "\n"
    (run_dir / "PROMPT.md").write_text(prompt)

    manifest = {
        "created_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": args.mode,
        "model": args.model,
        "prompt_version": args.version,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "prompt_sources": [str(p.relative_to(REPO)) for p in parts],
        "images": images,
        "repo_commit": git("rev-parse", "HEAD"),
        "repo_dirty": bool(git("status", "--porcelain")),
        "weights_lock_sha256": sha256(REPO / "weights.lock.json"),
        "uv_lock_sha256": sha256(REPO / "uv.lock"),
        "platform": platform.platform(),
        "libraries": library_versions(),
        "python_command": python_cmd,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    if manifest["repo_dirty"]:
        print("ATENÇÃO: há mudanças não commitadas; o commit no manifesto não descreve exatamente os prompts usados.")
    print(run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
