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
import fcntl
import json
import os
import pathlib
import shutil
import signal
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
# paradas antecipadas: uma sessao sem nenhuma requisicao ao modelo nesse tempo travou na inicializacao e e reiniciada
# (nao conta como tentativa do modelo); recusas seguidas do plugin sem nenhuma acao aceita entre elas encerram a sessao
# (laco de recusas); as mesmas pendencias em tantas tentativas seguidas encerram as retomadas (pendencia estagnada)
STARTUP_TIMEOUT_S = 60  # uma inicializacao saudavel faz a primeira requisicao em ate 5 s (p99 4,7 s em 247 casos)
STARTUP_RETRIES = 5
# muitas sessoes do opencode inicializando ao mesmo tempo travam (maquina carregada): no maximo tantas inicializacoes
# simultaneas em toda a maquina, do inicio do processo ate a primeira requisicao ao modelo
STARTUP_SLOTS = 4
REFUSAL_LOOP = 30
STALL_ATTEMPTS = 3
REFUSALS = {"blocked", "log_rejected", "path_blocked"}
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


def export_sessions(run_dir: pathlib.Path, ids: list[str], env: dict | None = None) -> bool:
    out = run_dir / "sessions"
    out.mkdir(exist_ok=True)
    ok = True
    for sid in ids:
        # direto para arquivo: por pipe, o opencode encerra antes de esvaziar a saída e o JSON sai truncado
        target = out / f"{sid}.json"
        with open(target, "w") as f:
            proc = subprocess.run(["opencode", "export", sid], stdout=f, stderr=subprocess.DEVNULL, env=env)
        if proc.returncode != 0:
            target.unlink()
            ok = False
    return ok


def harness_events(run_dir: pathlib.Path, offset: int) -> tuple[list[str], int]:
    """Tipos dos eventos novos no registro do plugin desde o byte offset; devolve também o novo offset."""
    log = run_dir / "harness_log.jsonl"
    if not log.exists():
        return [], offset
    with open(log, "rb") as f:
        f.seek(offset)
        data = f.read()
    end = data.rfind(b"\n") + 1  # só linhas completas
    kinds = []
    for line in data[:end].splitlines():
        try:
            kinds.append(json.loads(line).get("kind"))
        except json.JSONDecodeError:
            continue
    return kinds, offset + end


def startup_slot():
    """Trava uma das vagas de inicializacao (arquivos em .sandbox/startup-slots, compartilhados entre processos)."""
    slots = pathlib.Path(os.environ["SANDBOX"]) / "startup-slots"
    slots.mkdir(parents=True, exist_ok=True)
    while True:
        for i in range(STARTUP_SLOTS):
            f = open(slots / f"slot{i}", "w")
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return f
            except BlockingIOError:
                f.close()
        time.sleep(3)


def run_attempt(cmd: list[str], run_dir: pathlib.Path, env: dict, events: pathlib.Path, deadline: float) -> dict:
    """Uma sessão do opencode, vigiada: prazo total, travamento na inicialização e laço de recusas do plugin."""
    log = run_dir / "harness_log.jsonl"
    offset = log.stat().st_size if log.exists() else 0
    slot = startup_slot()
    begin, requests, streak = time.time(), 0, 0
    with open(events, "a") as out, open(run_dir / "opencode_stderr.log", "a") as err:
        # grupo de processos próprio: ao encerrar, nenhum filho do opencode fica para trás
        proc = subprocess.Popen(cmd, cwd=run_dir, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                start_new_session=True)
        stop = None
        while proc.poll() is None:
            time.sleep(5)
            kinds, offset = harness_events(run_dir, offset)
            for kind in kinds:
                requests += kind == "chat_params"
                streak = streak + 1 if kind in REFUSALS else (0 if kind == "tool_start" else streak)
            if requests and slot:
                slot.close()  # a sessao ja conversa com o modelo: libera a vaga de inicializacao
                slot = None
            if time.time() >= deadline:
                stop = "timed_out"
            elif not requests and time.time() - begin > STARTUP_TIMEOUT_S:
                stop = "startup_hang"
            elif streak >= REFUSAL_LOOP:
                stop = "refusal_loop"
            if stop:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
                break
    if slot:
        slot.close()
    return {"exit_code": None if stop else proc.returncode, "stopped": stop, "model_requests": requests}


def run_one(mode: str, model: str, image: pathlib.Path, provider: str, timeout_min: int, label: str,
            max_nudges: int, version: str = "v2", results_name: str | None = None, workers: int = 1) -> pathlib.Path | None:
    dest_mode_dir = EVAL / "results" / (results_name or version) / MODES[mode] / model
    if dest_mode_dir.exists():
        for existing in dest_mode_dir.glob(f"*-{label}"):
            # toda rodada concluida conta, inclusive as que falharam ou estouraram o tempo: refazer so as falhas
            # daria ao modelo novas chances nos casos dificeis e enviesaria o resultado
            if (existing / "harness_result.json").exists():
                print(f"[{time.strftime('%H:%M:%S')}] Pula {image.name}: rodada ja concluida em {existing.name}", flush=True)
                return existing

    # pausa coordenada com o servidor (por exemplo, para trocar a configuracao do modelo): nenhum caso novo comeca
    # enquanto existir .sandbox/pause/<modelo>; os casos em andamento terminam normalmente
    pause = pathlib.Path(os.environ["SANDBOX"]) / "pause" / model
    if pause.exists():
        print(f"[{time.strftime('%H:%M:%S')}] pausa: {image.name} espera {pause}", flush=True)
        while pause.exists():
            time.sleep(30)
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
    startup_restarts = 0
    message = prompt
    # banco do opencode próprio da rodada: com um banco comum, sessões abertas ao mesmo tempo disputam a trava e
    # algumas travam na inicialização; as sessões são exportadas para a rodada e o banco é apagado no fim
    data_home = pathlib.Path(os.environ["SANDBOX"]) / "opencode" / "runs" / model / run_dir.name
    data_home.mkdir(parents=True, exist_ok=True)
    # o opencode usa o PWD herdado, não o cwd do processo: sem isso a sessão abre no diretório do runner e o
    # opencode.json da rodada (subagente, permissões) não é carregado
    env = {**os.environ, "PWD": str(run_dir), "XDG_DATA_HOME": str(data_home)}
    while True:
        cmd = ["opencode", "run", "--model", f"{provider}/{model}", "--format", "json", "--auto", "--dir", str(run_dir)]
        session = summarize(events)["main_session"] if events.exists() else None
        cmd += ["--session", session] if session else ["--title", run_dir.name]
        attempt = run_attempt([*cmd, message], run_dir, env, events, deadline)
        if attempt["stopped"] == "startup_hang" and startup_restarts < STARTUP_RETRIES:
            # travou antes da primeira requisição ao modelo: falha de infraestrutura, a mesma mensagem é reenviada
            startup_restarts += 1
            print(f"    reinicio: sessao sem requisicao ao modelo em {STARTUP_TIMEOUT_S}s", flush=True)
            continue
        if attempt["stopped"] == "startup_hang":
            # a sessao (primeira ou retomada) nunca chegou ao modelo: falha de infraestrutura, nao do modelo. A rodada fica sem
            # harness_result.json, e a fila a refaz ao ser religada
            print(f"    INFRA: sessao travou na inicializacao {STARTUP_RETRIES + 1} vezes; rodada nao conta e sera refeita", flush=True)
            (run_dir / "infra_failure.json").write_text(json.dumps({"reason": "startup_hang", "restarts": startup_restarts}) + "\n")
            dest = EVAL / "results" / "descartadas-infra" / run_dir.relative_to(EVAL / "results")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(run_dir), dest)
            shutil.rmtree(data_home, ignore_errors=True)
            return None
        attempt["ended_at_s"] = round(time.time() - started, 1)
        if attempt["stopped"] == "timed_out":
            attempt["timed_out"] = True
        attempts.append(attempt)
        if attempt["stopped"] == "timed_out":
            break
        # a sessão terminou: o verificador decide se o contrato foi cumprido; se não, retoma com as pendências (a sessão
        # pode ter parado cedo, esquecido um arquivo ou emitido uma chamada de ferramenta como texto)
        passed, problems, _ = check(run_dir)
        attempts[-1].update({"check_passed": passed, "problems": sorted({reason for reason, _ in problems}),
                             "problem_texts": [text[:300] for _, text in problems],
                             "check_s": round(time.time() - started - attempts[-1]["ended_at_s"], 1)})
        if passed or not problems or attempt["stopped"] == "refusal_loop" or len(attempts) > max_nudges or time.time() >= deadline:
            break
        recent = [a.get("problem_texts") for a in attempts[-STALL_ATTEMPTS:]]
        if len(recent) == STALL_ATTEMPTS and all(r == recent[0] for r in recent):
            attempts[-1]["stopped"] = "stalled"
            break
        message = NUDGE.format(problems="\n".join(f"- {text}" for _, text in problems))
    elapsed = time.time() - started
    exit_code = attempts[-1].get("exit_code")
    timed_out = bool(attempts[-1].get("timed_out"))

    summary = summarize(events)
    if export_sessions(run_dir, [s for s in [summary["main_session"], *summary["child_sessions"]] if s], env):
        shutil.rmtree(data_home, ignore_errors=True)
    passed, problems, check_output = check(run_dir)
    nudged = attempts[:-1]
    result = {
        "exit_code": exit_code,
        "timed_out": timed_out,
        "nudges": len(nudged),
        # retomadas por motivo (uma retomada pode ter varios): final_json, required_file, tools_json, reproduce, inspection
        "nudges_by_reason": {r: sum(r in a.get("problems", []) for a in nudged) for r in sorted({r for a in nudged for r in a.get("problems", [])})},
        "contract_first_attempt": bool(attempts[0].get("check_passed")),
        # parada antecipada da ultima tentativa: refusal_loop, stalled, startup_hang ou timed_out (vazio se terminou sozinha)
        "early_stop": attempts[-1].get("stopped") or "",
        "startup_restarts": startup_restarts,
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
        # falha de infraestrutura (sessao que nunca chegou ao modelo): o caso e refeito do zero, ate 3 vezes
        for _ in range(3):
            done = run_one(args.mode, args.model, image.resolve(), args.provider, args.timeout, label, effective_nudges,
                           args.version, results_name, args.workers)
            if done:
                return done
            time.sleep(120)
        print(f"[{time.strftime('%H:%M:%S')}] INFRA: {label} desistiu depois de 3 falhas de infraestrutura", flush=True)

    # cada caso roda em processos opencode próprios; as threads só esperam por eles
    with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as pool:
        futures = {pool.submit(run, job): job for job in jobs}
        for future in as_completed(futures):
            if future.exception():
                print(f"[{time.strftime('%H:%M:%S')}] ERRO em {futures[future][1]}: {future.exception()!r}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
