"""Roda a bancada no opencode, sem humano no loop, uma sessão por imagem.

Uso:
    uv run python tools/run_opencode.py --mode single --model qwen3.8-27b
    uv run python tools/run_opencode.py --mode agents --model gemma4-26b --images inputs/ID003-xray.png
    uv run python tools/run_opencode.py --mode single --model qwen3.6-27b --repeat 3

Para cada imagem: cria a rodada (tools/new_run.py, com as notas de harness do opencode), grava um
`opencode.json` local que fixa o mesmo modelo para o agente principal e para os subagentes e nega acesso
a diretórios de fora, executa `opencode run --format json` e salva na rodada:

  opencode_events.jsonl   eventos da sessão principal
  sessions/<id>.json      exportação completa da sessão principal e de cada subagente
  harness_result.json     código de saída, duração, tokens, contagem de chamadas de ferramenta
  check_report.json       resultado de tools/check_run.py

O provedor (`DGX-UFSC`, com URL e chave) vem da configuração global do opencode do usuário, para a
chave não entrar no repositório.
"""

import argparse
import json
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from new_run import BENCH, MODELS_DIR, create_run  # noqa: E402

AGENT_STEPS = 400


def opencode_config(provider: str, model: str, run_dir: pathlib.Path) -> dict:
    import torchxrayvision

    full = f"{provider}/{model}"
    txv = pathlib.Path(torchxrayvision.__file__).parent
    return {
        "$schema": "https://opencode.ai/config.json",
        "model": full,
        "small_model": full,
        "autoupdate": False,
        "share": "disabled",
        "agent": {
            "build": {"model": full, "steps": AGENT_STEPS},
            "analyst": {
                "mode": "subagent",
                "model": full,
                "steps": AGENT_STEPS,
                "description": "Independent radiograph analyst. Runs the same model as the orchestrator. Use for every sub-agent of the council.",
                "prompt": "You are one independent analyst of a single-model council. Follow the task description exactly, write the files it asks for, and report back briefly.",
            },
            "general": {"model": full},
            "explore": {"model": full},
        },
        "permission": {
            "question": "deny",
            "webfetch": "deny",
            "websearch": "deny",
            "external_directory": {
                "*": "deny",
                f"{BENCH / 'tools'}/*": "allow",
                f"{txv}/*": "allow",
                f"{MODELS_DIR}/*": "allow",
                "/tmp/*": "allow",
            },
        },
    }


def summarize(events_path: pathlib.Path) -> dict:
    tokens = {"input": 0, "output": 0, "reasoning": 0}
    tools: dict[str, int] = {}
    child_sessions, main_session = [], None
    for line in events_path.read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        main_session = main_session or event.get("sessionID")
        part = event.get("part", {})
        if event.get("type") == "step_finish":
            for key in tokens:
                tokens[key] += part.get("tokens", {}).get(key, 0)
        if event.get("type") == "tool_use":
            tools[part.get("tool")] = tools.get(part.get("tool"), 0) + 1
            child = part.get("state", {}).get("metadata", {}).get("sessionId")
            if child:
                child_sessions.append(child)
    return {"main_session": main_session, "child_sessions": child_sessions, "tokens_main_session": tokens, "tool_calls": tools}


def export_sessions(run_dir: pathlib.Path, ids: list[str]) -> None:
    out = run_dir / "sessions"
    out.mkdir(exist_ok=True)
    for sid in ids:
        proc = subprocess.run(["opencode", "export", sid], capture_output=True, text=True)
        if proc.returncode == 0:
            (out / f"{sid}.json").write_text(proc.stdout)


def run_one(mode: str, model: str, image: pathlib.Path, provider: str, timeout_min: int, label: str) -> pathlib.Path:
    run_dir = create_run(mode, model, [image], harness="opencode", label=label)
    (run_dir / "opencode.json").write_text(json.dumps(opencode_config(provider, model, run_dir), indent=2) + "\n")
    prompt = (run_dir / "PROMPT.md").read_text()

    print(f"[{time.strftime('%H:%M:%S')}] {mode} {model} {image.name} -> {run_dir}", flush=True)
    started = time.time()
    with open(run_dir / "opencode_events.jsonl", "w") as out, open(run_dir / "opencode_stderr.log", "w") as err:
        try:
            proc = subprocess.run(
                ["opencode", "run", "--model", f"{provider}/{model}", "--format", "json", "--auto",
                 "--title", run_dir.name, prompt],
                cwd=run_dir, stdout=out, stderr=err, timeout=timeout_min * 60,
            )
            exit_code, timed_out = proc.returncode, False
        except subprocess.TimeoutExpired:
            exit_code, timed_out = None, True
    elapsed = time.time() - started

    summary = summarize(run_dir / "opencode_events.jsonl")
    export_sessions(run_dir, [s for s in [summary["main_session"], *summary["child_sessions"]] if s])
    check = subprocess.run([sys.executable, str(BENCH / "tools" / "check_run.py"), str(run_dir)], capture_output=True, text=True)
    result = {
        "exit_code": exit_code,
        "timed_out": timed_out,
        "elapsed_s": round(elapsed, 1),
        "check_passed": check.returncode == 0,
        "check_output": check.stdout[-4000:],
        **summary,
    }
    (run_dir / "harness_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"    {elapsed / 60:.1f} min, exit={exit_code}, timeout={timed_out}, check={'ok' if result['check_passed'] else 'FALHA'}", flush=True)
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "agents"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider", default="DGX-UFSC")
    parser.add_argument("--images", nargs="+", type=pathlib.Path, default=sorted((BENCH / "inputs").glob("*.png")))
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=120, help="minutos por sessão")
    args = parser.parse_args()

    for rep in range(1, args.repeat + 1):
        for image in args.images:
            label = f"{image.stem.split('-')[0]}-r{rep}"
            run_one(args.mode, args.model, image.resolve(), args.provider, args.timeout, label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
