"""Prepara um diretório de rodada isolado, com o prompt renderizado e um manifesto de proveniência.

Uso:
    tools/py tools/new_run.py --mode single --model sonnet5
    tools/py tools/new_run.py --mode agents --model qwen3.8-27b --images inputs/ID003-xray.png

Cria `<root>/<versão>/<modo>/<modelo>/<timestamp>/` com cópias das imagens, `PROMPT.md` (o texto exato
a ser entregue ao agente) e `run_manifest.json` (hashes do prompt, das imagens, dos pesos, commit do
repositório e versões das bibliotecas). Depois é só abrir o agente nesse diretório e passar o PROMPT.md.
Para rodar automaticamente no opencode, use tools/run_opencode.py.
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

EVAL = pathlib.Path(__file__).resolve().parent.parent
PROMPTS = EVAL / "prompts"
MODES = {"single": "single-agent", "agents": "multi-agent-single-model"}
MODELS_DIR = EVAL / "models"
PYTHON = EVAL / "tools" / "py"


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=EVAL, capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def library_versions() -> dict:
    import importlib.metadata as md

    names = ["numpy", "scipy", "opencv-python-headless", "scikit-image", "pillow", "pywavelets", "simpleitk",
             "torch", "torchvision", "timm", "transformers", "open_clip_torch", "ultralytics", "torchxrayvision"]
    return {"python": platform.python_version(), **{n: md.version(n) for n in names}}


def render(template: str, values: dict) -> str:
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", value)
    if "{{" in template:
        raise SystemExit("placeholder sem valor no prompt renderizado")
    return template


def prompt_parts(version: str, mode: str, harness: str | None) -> list[pathlib.Path]:
    base = PROMPTS / version
    parts = [base / f"{mode}.md", *sorted((base / "_shared").glob("*.md"))]
    if harness:
        parts.append(base / "_harness" / f"{harness}.md")
        extra = base / "_harness" / f"{harness}-{mode}.md"
        if extra.exists():
            parts.append(extra)
    return parts


def create_run(mode: str, model: str, images: list[pathlib.Path], version: str = "v2",
               root: pathlib.Path = EVAL / "results", harness: str | None = None, label: str = "") -> pathlib.Path:
    # confere os pesos antes de gastar uma rodada com um ambiente divergente
    verify = subprocess.run([sys.executable, str(EVAL / "tools" / "setup_models.py"), "--verify"], capture_output=True, text=True)
    if verify.returncode != 0:
        print(verify.stdout, file=sys.stderr)
        raise SystemExit("pesos ausentes ou divergentes; rode tools/setup_models.py")

    import torchxrayvision

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + (f"-{label}" if label else "")
    run_dir = (root / version / MODES[mode] / model / stamp).resolve()
    run_dir.mkdir(parents=True)

    copied = []
    for image in images:
        shutil.copy2(image, run_dir / image.name)
        copied.append({"file": image.name, "sha256": sha256(image)})

    values = {
        "MODEL_NAME": model,
        "IMAGES": ", ".join(f"`{i['file']}`" for i in copied),
        "RUN_DIR": str(run_dir),
        "PYTHON": str(PYTHON),
        "MODELS_DIR": str(MODELS_DIR),
        "TXV_SOURCE": str(pathlib.Path(torchxrayvision.__file__).parent),
        "TXV_VERSION": torchxrayvision.__version__,
    }
    parts = prompt_parts(version, mode, harness)
    prompt = "\n\n".join(render(p.read_text(), values).strip() for p in parts) + "\n"
    (run_dir / "PROMPT.md").write_text(prompt)

    manifest = {
        "created_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": mode,
        "model": model,
        "harness": harness,
        "prompt_version": version,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "prompt_sources": [str(p.relative_to(EVAL)) for p in parts],
        "images": copied,
        "repo_commit": git("rev-parse", "HEAD"),
        "repo_dirty": bool(git("status", "--porcelain", "--", ".")),
        "models_lock_sha256": sha256(EVAL / "models.lock.json"),
        "uv_lock_sha256": sha256(EVAL / "uv.lock"),
        "platform": platform.platform(),
        "libraries": library_versions(),
        "python_command": str(PYTHON),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    if manifest["repo_dirty"]:
        print("ATENÇÃO: há mudanças não commitadas; o commit no manifesto não descreve exatamente os prompts usados.", file=sys.stderr)
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--model", required=True, help="nome do modelo avaliado, ex.: sonnet5, qwen3.8-27b")
    parser.add_argument("--version", default="v2", help="versão dos prompts em prompts/<versão>/")
    parser.add_argument("--images", nargs="+", type=pathlib.Path, default=sorted((EVAL / "inputs").glob("*.png")))
    parser.add_argument("--root", type=pathlib.Path, default=EVAL / "results", help="onde criar a rodada (dentro do projeto)")
    parser.add_argument("--harness", help="acrescenta prompts/<versão>/_harness/<harness>.md, ex.: opencode")
    args = parser.parse_args()

    print(create_run(args.mode, args.model, args.images, args.version, args.root, args.harness))
    return 0


if __name__ == "__main__":
    sys.exit(main())
