"""Roda a bancada no opencode, sem humano no loop, uma sessão por imagem.

Uso:
    tools/py tools/run_opencode.py --mode single --model qwen3.8-27b
    tools/py tools/run_opencode.py --mode agents --model gemma4-26b --images inputs/ID003-xray.png
    tools/py tools/run_opencode.py --mode single --model qwen3.6-27b --repeat 3

Para cada imagem: cria a rodada (tools/new_run.py, com as notas de harness do opencode), grava um
`opencode.json` local que fixa o mesmo modelo para o agente principal e para os subagentes e nega acesso
a diretórios de fora, executa `opencode run --format json` e salva na rodada:

  opencode_events.jsonl   eventos da sessão principal
  sessions/<id>.json      exportação completa da sessão principal e de cada subagente
  harness_result.json     código de saída, duração, tokens, chamadas de ferramenta e retomadas

Se a sessão termina sem o JSON final (por exemplo, quando o modelo emite a chamada de ferramenta como
texto e o servidor não a converte), o runner retoma a mesma sessão com uma mensagem neutra, até
--max-nudges vezes; o número de retomadas é registrado e é uma métrica da rodada.
  check_report.json       resultado de tools/check_run.py

O provedor (`DGX-UFSC`) está em harness/opencode/opencode.json, com URL e chave lidas do .env local.
Como roda sob tools/py, o opencode herda o env.sh: configuração, sessões, cache e HOME ficam em
.sandbox/, e nada é gravado fora do projeto.
"""

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from new_run import EVAL, MODELS_DIR, create_run  # noqa: E402

AGENT_STEPS = 400
# mensagem de retomada quando a sessão termina sem o JSON final; neutra quanto ao conteúdo da análise
NUDGE = (
    "The task is not complete yet: `{json}` does not exist. If your previous message contained a tool call "
    "written as plain text, it was not executed. Continue from where you stopped and finish every step of the task."
)


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
                f"{EVAL / 'tools'}/*": "allow",
                f"{txv}/*": "allow",
                f"{MODELS_DIR}/*": "allow",
                f"{os.environ['TMPDIR']}/*": "allow",
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


def run_one(mode: str, model: str, image: pathlib.Path, provider: str, timeout_min: int, label: str,
            max_nudges: int) -> pathlib.Path:
    run_dir = create_run(mode, model, [image], harness="opencode", label=label)
    (run_dir / "opencode.json").write_text(json.dumps(opencode_config(provider, model, run_dir), indent=2) + "\n")
    prompt = (run_dir / "PROMPT.md").read_text()

    print(f"[{time.strftime('%H:%M:%S')}] {mode} {model} {image.name} -> {run_dir}", flush=True)
    final_json = run_dir / f"{image.stem}_analysis" / f"{image.name}.json"
    events = run_dir / "opencode_events.jsonl"
    started = time.time()
    deadline = started + timeout_min * 60
    attempts = []
    message = prompt
    while True:
        cmd = ["opencode", "run", "--model", f"{provider}/{model}", "--format", "json", "--auto"]
        session = summarize(events)["main_session"] if events.exists() else None
        cmd += ["--session", session] if session else ["--title", run_dir.name]
        with open(events, "a") as out, open(run_dir / "opencode_stderr.log", "a") as err:
            try:
                proc = subprocess.run([*cmd, message], cwd=run_dir, stdout=out, stderr=err,
                                      timeout=max(deadline - time.time(), 1))
                attempts.append({"exit_code": proc.returncode, "ended_at_s": round(time.time() - started, 1)})
            except subprocess.TimeoutExpired:
                attempts.append({"exit_code": None, "timed_out": True, "ended_at_s": round(time.time() - started, 1)})
                break
        # a sessão terminou sem o JSON final: pode ter sido uma chamada de ferramenta emitida como texto
        # (o servidor não a converteu) ou uma parada prematura; retoma com uma mensagem neutra
        if final_json.exists() or len(attempts) > max_nudges or time.time() >= deadline:
            break
        message = NUDGE.format(json=final_json.relative_to(run_dir))
    elapsed = time.time() - started
    exit_code = attempts[-1].get("exit_code")
    timed_out = bool(attempts[-1].get("timed_out"))

    summary = summarize(events)
    export_sessions(run_dir, [s for s in [summary["main_session"], *summary["child_sessions"]] if s])
    check = subprocess.run([sys.executable, str(EVAL / "tools" / "check_run.py"), str(run_dir)], capture_output=True, text=True)
    result = {
        "exit_code": exit_code,
        "timed_out": timed_out,
        "nudges": len(attempts) - 1,
        "attempts": attempts,
        "final_json_written": final_json.exists(),
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
    parser.add_argument("--images", nargs="+", type=pathlib.Path, default=sorted((EVAL / "inputs").glob("*.png")))
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=120, help="minutos por sessão, somando as retomadas")
    parser.add_argument("--max-nudges", type=int, default=5, help="retomadas automáticas se faltar o JSON final")
    args = parser.parse_args()

    for rep in range(1, args.repeat + 1):
        for image in args.images:
            label = f"{image.stem.split('-')[0]}-r{rep}"
            run_one(args.mode, args.model, image.resolve(), args.provider, args.timeout, label, args.max_nudges)
    return 0


if __name__ == "__main__":
    sys.exit(main())
