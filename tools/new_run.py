"""Prepara um diretório de rodada isolado, com o prompt renderizado e um manifesto de proveniência.

Uso:
    tools/py tools/new_run.py --mode single --model qwen3.6-27b
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
HARNESS_TEMPLATE = EVAL / "harness" / "opencode" / "run_template"
AGENT_STEPS = 400
# versoes em que o harness do opencode e gerado pela criacao da rodada (AGENTS.md, .opencode/, plugin)
NATIVE_HARNESS = {"v4", "v4c"}  # v4c: vocabulario fechado, gerado por tools/make_closed_prompts.py


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
    # o prompt multiagente da v1 tinha o nome do modelo para ser preenchido à mão
    template = template.replace("<MODEL HERE>", values["MODEL_NAME"])
    if "{{" in template:
        raise SystemExit("placeholder sem valor no prompt renderizado")
    return template


def prompt_parts(version: str, mode: str, harness: str | None, results_name: str | None = None) -> list[pathlib.Path]:
    base = PROMPTS / version
    exp2_candidate = base / f"{mode}_exp2.md"
    mode_file = exp2_candidate if (results_name and "exp2" in results_name and exp2_candidate.exists()) else (base / f"{mode}.md")
    parts = [mode_file, *sorted((base / "_shared").glob("*.md"))]  # a v1 nao tem _shared
    if harness:
        parts.append(base / "_harness" / f"{harness}.md")
        extra = base / "_harness" / f"{harness}-{mode}.md"
        if extra.exists():
            parts.append(extra)
    return parts


def harness_opencode_config(provider: str, model: str, run_dir: pathlib.Path) -> dict:
    """opencode.json da rodada (v4): o agente analyst vem de .opencode/agent/analyst.md."""
    import os

    full = f"{provider}/{model}"
    results = EVAL / "results"
    # arquivos do harness que o agente nao pode alterar: o log so e escrito pela ferramenta log_action
    protected = ["*agent_log*", "*harness_log*", "*harness.json*", "*.opencode*", "*opencode.json*", "*AGENTS.md*"]
    return {
        "$schema": "https://opencode.ai/config.json",
        "model": full,
        "small_model": full,
        "autoupdate": False,
        "share": "disabled",
        # sem snapshots git por passo: sao lentos e nao fazem parte da tarefa
        "snapshot": False,
        "agent": {"build": {"model": full, "steps": AGENT_STEPS}, "general": {"model": full}, "explore": {"model": full}},
        # a ultima regra que casa vence: o geral primeiro, as excecoes depois
        "permission": {
            "question": "deny",
            "webfetch": "deny",
            "websearch": "deny",
            "external_directory": {"*": "deny", "/tmp/*": "allow"},
            # dentro do repositorio, os resultados de outras rodadas contaminariam a analise
            **{tool: {"*": "allow", f"*{results}*": "deny", f"{results}/**": "deny", f"{results}/*": "deny",
                      f"{run_dir}/*": "allow", f"{run_dir}/**": "allow"} for tool in ["read", "list", "glob"]},
            "edit": {"*": "deny", f"{run_dir}/*": "allow", f"{run_dir}/**": "allow", f"{os.environ['TMPDIR']}/*": "allow",
                     "/tmp/*": "allow", **{p: "deny" for p in protected}},
            "bash": {"*": "allow", f"*{results}*": "deny", "*../*": "deny", f"*{run_dir}*": "allow", **{p: "deny" for p in protected}},
        },
    }


def write_native_harness(run_dir: pathlib.Path, version: str, mode: str, model: str, images: list[dict],
                         values: dict, results_name: str | None) -> list[pathlib.Path]:
    """Copia o modelo .opencode/ e escreve AGENTS.md, opencode.json e harness.json da rodada."""
    base = PROMPTS / version
    for src in HARNESS_TEMPLATE.rglob("*"):
        if src.is_file():
            dst = run_dir / src.relative_to(HARNESS_TEMPLATE)
            dst.parent.mkdir(parents=True, exist_ok=True)
            text = src.read_text()
            dst.write_text(render(text, values) if src.suffix == ".md" else text)
    shared = [*sorted((base / "_shared").glob("*.md")), base / "_harness" / "opencode.md"]
    (run_dir / "AGENTS.md").write_text("\n\n".join(render(p.read_text(), values).strip() for p in shared) + "\n")
    (run_dir / "opencode.json").write_text(json.dumps(harness_opencode_config("DGX-UFSC", model, run_dir), indent=2) + "\n")
    stem = pathlib.Path(images[0]["file"]).stem
    (run_dir / "harness.json").write_text(json.dumps({
        "analysis_dir": f"{stem}_analysis",
        "agent_log": f"{stem}_analysis/provenance/agent_log.jsonl",
        "harness_log": "harness_log.jsonl",
        "mode": mode,
    }, indent=2) + "\n")
    return shared


def create_run(mode: str, model: str, images: list[pathlib.Path], version: str = "v2",
               root: pathlib.Path = EVAL / "results", harness: str | None = None, label: str = "",
               results_name: str | None = None) -> pathlib.Path:
    # confere os pesos antes de gastar uma rodada com um ambiente divergente
    verify = subprocess.run([sys.executable, str(EVAL / "tools" / "setup_models.py"), "--verify"], capture_output=True, text=True)
    if verify.returncode != 0:
        print(verify.stdout, file=sys.stderr)
        raise SystemExit("pesos ausentes ou divergentes; rode tools/setup_models.py")

    import torchxrayvision

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + (f"-{label}" if label else "")
    # results_name separa condições com o mesmo prompt, ex.: v1-runner (v1 no runner) e v2-ablation
    run_dir = (root / (results_name or version) / MODES[mode] / model / stamp).resolve()
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
    parts = prompt_parts(version, mode, harness, results_name=results_name)
    shared_parts: list[pathlib.Path] = []
    if harness == "opencode" and version in NATIVE_HARNESS:
        # na v4 as regras comuns vao para AGENTS.md (todo agente recebe); o PROMPT.md fica so com a tarefa
        shared_parts = write_native_harness(run_dir, version, mode, model, copied, values, results_name)
        parts = [p for p in parts if p not in shared_parts]
    prompt = "\n\n".join(render(p.read_text(), values).strip() for p in parts) + "\n"
    (run_dir / "PROMPT.md").write_text(prompt)
    harness_files = sorted(p for p in run_dir.rglob("*") if p.is_file() and p.name != "PROMPT.md"
                           and (p.name in ("AGENTS.md", "opencode.json", "harness.json") or ".opencode" in p.parts))

    manifest = {
        "created_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": mode,
        "model": model,
        "harness": harness,
        "prompt_version": version,
        "condition": results_name or version,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "prompt_sources": [str(p.relative_to(EVAL)) for p in parts + shared_parts],
        "harness_files": {str(p.relative_to(run_dir)): sha256(p) for p in harness_files},
        "images": copied,
        "repo_commit": git("rev-parse", "HEAD"),
        "repo_dirty": bool(git("status", "--porcelain", "--", ".", ":!results")),  # resultados em andamento não contam
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
    parser.add_argument("--model", required=True, help="nome do modelo avaliado, ex.: qwen3.6-27b, qwen3.8-27b, gemma4-26b")
    parser.add_argument("--version", default="v2", help="versão dos prompts em prompts/<versão>/")
    parser.add_argument("--images", nargs="+", type=pathlib.Path, default=sorted((EVAL / "inputs").glob("*.png")))
    parser.add_argument("--root", type=pathlib.Path, default=EVAL / "results", help="onde criar a rodada (dentro do projeto)")
    parser.add_argument("--harness", help="acrescenta prompts/<versão>/_harness/<harness>.md, ex.: opencode")
    args = parser.parse_args()

    print(create_run(args.mode, args.model, args.images, args.version, args.root, args.harness))
    return 0


if __name__ == "__main__":
    sys.exit(main())
