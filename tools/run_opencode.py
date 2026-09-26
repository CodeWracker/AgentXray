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

  check_report.json       resultado de tools/check_run.py

O harness garante o contrato da rodada: toda vez que a sessão termina, o runner roda tools/check_run.py (JSON
final, arquivos exigidos, tools.json, reprodução byte a byte e inspeção visual das imagens geradas). Se algo
falhou, retoma a mesma sessão com a lista exata das pendências, sem nada sobre o conteúdo da análise, até
--max-nudges vezes ou o fim do tempo. Cada retomada fica registrada com os motivos; a conformidade na primeira
tentativa, o número de retomadas e os motivos são métricas da rodada.

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
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from check_run import agent_problems  # noqa: E402
from new_run import EVAL, create_run, MODES  # noqa: E402

AGENT_STEPS = 400
# mensagem de retomada quando a sessão termina sem cumprir o contrato; só aponta pendências verificáveis (arquivos,
# reprodução, inspeção) e nada sobre o conteúdo da análise
NUDGE = (
    "The task is not complete yet. An automated check of your run directory found these problems:\n\n{problems}\n\n"
    "If your previous message contained a tool call written as plain text, it was not executed. Fix every problem "
    "listed, keeping the analysis you already did, and finish every remaining step of the task."
)
# eventos do plugin contados em harness_result.json
PLUGIN_EVENTS = {"blocked": "unlogged_action_blocks", "path_blocked": "path_blocks", "log_rejected": "log_rejections"}


def opencode_config(provider: str, model: str, run_dir: pathlib.Path) -> dict:
    full = f"{provider}/{model}"
    results = EVAL / "results"
    return {
        "$schema": "https://opencode.ai/config.json",
        "model": full,
        "small_model": full,
        "autoupdate": False,
        "share": "disabled",
        # sem snapshots git por passo: são lentos e não fazem parte da tarefa
        "snapshot": False,
        "agent": {
            "build": {"model": full, "steps": AGENT_STEPS},
            "analyst": {
                "mode": "subagent",
                "model": full,
                "steps": AGENT_STEPS,
                "description": "Independent radiograph analyst. Runs the same model as the orchestrator. Use for every sub-agent of the council.",
                "prompt": "You are one independent analyst of a single-model council. Follow the task description exactly, write the files it asks for, keep your own agent log as described in the AGENT LOG section of PROMPT.md, and report back briefly.",
            },
            "general": {"model": full},
            "explore": {"model": full},
        },
        # a última regra que casa vence: o geral primeiro, as exceções depois
        "permission": {
            "question": "deny",
            "webfetch": "deny",
            "websearch": "deny",
            # fora do repositório da avaliação só o que a tarefa precisa
            "external_directory": {"*": "deny", "/tmp/*": "allow"},
            # dentro do repositório, os resultados de outras rodadas contaminariam a análise
            **{tool: {"*": "allow", f"*{results}*": "deny", f"{results}/**": "deny", f"{results}/*": "deny", f"{run_dir}/*": "allow", f"{run_dir}/**": "allow"} for tool in ["read", "list", "glob"]},
            "edit": {"*": "deny", f"{run_dir}/*": "allow", f"{os.environ['TMPDIR']}/*": "allow", "/tmp/*": "allow"},
            "bash": {"*": "allow", f"*{results}*": "deny", "*../*": "deny", f"*{run_dir}*": "allow"},
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


def check(run_dir: pathlib.Path) -> tuple[bool, list[tuple[str, str]], str]:
    """Roda o verificador na rodada; devolve se passou, as pendências para o agente e a saída do verificador."""
    proc = subprocess.run([sys.executable, str(EVAL / "tools" / "check_run.py"), str(run_dir)], capture_output=True, text=True)
    report_path = run_dir / "check_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    problems = [p for entry in report.values() for p in agent_problems(entry)]
    if proc.returncode != 0 and not problems:  # falha que o agente não pode corrigir (log) ou pasta de análise ausente
        problems = [] if report else [("final_json", "The analysis folder and the final JSON do not exist.")]
    return proc.returncode == 0, problems, proc.stdout


def plugin_counts(run_dir: pathlib.Path) -> dict:
    counts = dict.fromkeys(PLUGIN_EVENTS.values(), 0)
    log = run_dir / "harness_log.jsonl"
    if log.exists():
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                kind = json.loads(line).get("kind")
            except json.JSONDecodeError:
                continue
            if kind in PLUGIN_EVENTS:
                counts[PLUGIN_EVENTS[kind]] += 1
    return counts


def export_sessions(run_dir: pathlib.Path, ids: list[str]) -> None:
    out = run_dir / "sessions"
    out.mkdir(exist_ok=True)
    for sid in ids:
        # direto para arquivo: por pipe, o opencode encerra antes de esvaziar a saída e o JSON sai truncado
        target = out / f"{sid}.json"
        with open(target, "w") as f:
            proc = subprocess.run(["opencode", "export", sid], stdout=f, stderr=subprocess.DEVNULL)
        if proc.returncode != 0:
            target.unlink()


def run_one(mode: str, model: str, image: pathlib.Path, provider: str, timeout_min: int, label: str,
            max_nudges: int, version: str = "v2", results_name: str | None = None, workers: int = 1) -> pathlib.Path:
    dest_mode_dir = EVAL / "results" / (results_name or version) / MODES[mode] / model
    if dest_mode_dir.exists():
        for existing in dest_mode_dir.glob(f"*-{label}"):
            if (existing / "harness_result.json").exists():
                try:
                    h_info = json.loads((existing / "harness_result.json").read_text(encoding="utf-8"))
                    if h_info.get("check_passed") or h_info.get("final_json_written"):
                        print(f"[{time.strftime('%H:%M:%S')}] Pula {image.name}: rodada ja concluida em {existing.name}", flush=True)
                        return existing
                except Exception:
                    pass

    run_dir = create_run(mode, model, [image], version=version, harness="opencode", label=label,
                         results_name=results_name)
    if not (run_dir / "opencode.json").exists():  # na v4 o new_run ja escreveu o harness completo da rodada
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
        cmd = ["opencode", "run", "--model", f"{provider}/{model}", "--format", "json", "--auto", "--dir", str(run_dir)]
        session = summarize(events)["main_session"] if events.exists() else None
        cmd += ["--session", session] if session else ["--title", run_dir.name]
        with open(events, "a") as out, open(run_dir / "opencode_stderr.log", "a") as err:
            try:
                # o opencode usa o PWD herdado, não o cwd do processo: sem isso a sessão abre no diretório
                # do runner e o opencode.json da rodada (subagente, permissões) não é carregado
                # sem stdin: o opencode run le a entrada padrao quando ela nao e um terminal e esperaria para sempre
                proc = subprocess.run([*cmd, message], cwd=run_dir, env={**os.environ, "PWD": str(run_dir)},
                                      stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                      timeout=max(deadline - time.time(), 1))
                attempts.append({"exit_code": proc.returncode, "ended_at_s": round(time.time() - started, 1)})
            except subprocess.TimeoutExpired:
                attempts.append({"exit_code": None, "timed_out": True, "ended_at_s": round(time.time() - started, 1)})
                break
        # a sessão terminou: o verificador decide se o contrato foi cumprido; se não, retoma com as pendências (a sessão
        # pode ter parado cedo, esquecido um arquivo ou emitido uma chamada de ferramenta como texto)
        passed, problems, _ = check(run_dir)
        attempts[-1].update({"check_passed": passed, "problems": sorted({reason for reason, _ in problems}),
                             "check_s": round(time.time() - started - attempts[-1]["ended_at_s"], 1)})
        if passed or not problems or len(attempts) > max_nudges or time.time() >= deadline:
            break
        message = NUDGE.format(problems="\n".join(f"- {text}" for _, text in problems))
    elapsed = time.time() - started
    exit_code = attempts[-1].get("exit_code")
    timed_out = bool(attempts[-1].get("timed_out"))

    summary = summarize(events)
    export_sessions(run_dir, [s for s in [summary["main_session"], *summary["child_sessions"]] if s])
    passed, problems, check_output = check(run_dir)
    nudged = attempts[:-1]
    result = {
        "exit_code": exit_code,
        "timed_out": timed_out,
        "nudges": len(nudged),
        # retomadas por motivo (uma retomada pode ter varios): final_json, required_file, tools_json, reproduce, inspection
        "nudges_by_reason": {r: sum(r in a.get("problems", []) for a in nudged) for r in sorted({r for a in nudged for r in a.get("problems", [])})},
        "contract_first_attempt": bool(attempts[0].get("check_passed")),
        "attempts": attempts,
        "final_json_written": final_json.exists(),
        "elapsed_s": round(elapsed, 1),
        # parte do tempo de parede gasta pelo verificador entre as tentativas (reprodução em cópia limpa)
        "checker_s": round(sum(a.get("check_s", 0) for a in attempts), 1),
        # casos rodando ao mesmo tempo no mesmo modelo: o tempo de parede so e comparavel entre rodadas de mesmo valor
        "concurrent_workers": workers,
        "check_passed": passed,
        "remaining_problems": sorted({reason for reason, _ in problems}),
        "check_output": check_output[-4000:],
        **plugin_counts(run_dir),
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
    parser.add_argument("--images", nargs="+", type=pathlib.Path, default=None)
    parser.add_argument("--manifest", type=pathlib.Path, help="caminho para manifest.json de benchmark")
    parser.add_argument("--max-cases", type=int, default=None, help="limite maximo de casos a rodar do manifest")
    parser.add_argument("--case-range", nargs=2, type=int, metavar=("START", "END"), help="fatia de indices [START, END) do manifest")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=120, help="minutos por sessão, somando as retomadas")
    parser.add_argument("--max-nudges", type=int, default=None, help="retomadas automáticas enquanto o contrato não for cumprido")
    parser.add_argument("--version", default="v2", help="versão dos prompts (v1 ou v2)")
    parser.add_argument("--results-name", help="pasta da condição em results/ (padrão: v2, ou v1-runner para a v1)")
    parser.add_argument("--workers", type=int, default=1, help="casos simultâneos no mesmo modelo (registrado em harness_result.json)")
    args = parser.parse_args()

    if args.manifest:
        manifest_file = args.manifest.resolve()
        with open(manifest_file, encoding="utf-8") as f:
            items = json.load(f)
        if args.case_range:
            start_idx, end_idx = args.case_range
            items = items[start_idx:end_idx]
        elif args.max_cases:
            items = items[:args.max_cases]
        nih_images = EVAL / "inputs" / "nih_chestxray" / "images"
        images = []
        for it in items:
            img_name = it["image"]
            p = nih_images / img_name
            if not p.exists():
                candidates = list(EVAL.glob(f"**/{img_name}"))
                if candidates:
                    p = candidates[0]
            images.append(p)
        bench_dir_name = manifest_file.parent.name
        results_name = args.results_name or f"{args.version}/{bench_dir_name}"
    else:
        images = args.images or sorted((EVAL / "inputs").glob("*.png"))
        results_name = args.results_name or ("v1-runner" if args.version == "v1" else None)

    effective_nudges = args.max_nudges if args.max_nudges is not None else (40 if args.mode == "agents" else 8)
    jobs = [(image, f"{image.stem.split('-')[0]}-r{rep}") for rep in range(1, args.repeat + 1) for image in images]

    def run(job):
        image, label = job
        return run_one(args.mode, args.model, image.resolve(), args.provider, args.timeout, label, effective_nudges,
                       args.version, results_name, args.workers)

    # cada caso roda em processos opencode próprios; as threads só esperam por eles
    with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as pool:
        futures = {pool.submit(run, job): job for job in jobs}
        for future in as_completed(futures):
            if future.exception():
                print(f"[{time.strftime('%H:%M:%S')}] ERRO em {futures[future][1]}: {future.exception()!r}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
