"""Autoteste do harness do opencode: confere, com o modelo de verdade, o que o runner promete.

Uso:
    tools/py tools/harness_selftest.py --model qwen3.8-27b [--model gemma4-26b ...]

Para cada modelo, monta um diretório descartável em $TMPDIR com o mesmo opencode.json das rodadas, pede
ao agente três ações e verifica na sessão exportada:
  1. a sessão abriu no diretório da rodada (senão o opencode.json da rodada não é carregado);
  2. ler um arquivo de fora do diretório foi negado pela permissão;
  3. o subagente `analyst` existe e roda o mesmo modelo;
  4. o intérprete do projeto (tools/py) funciona de dentro da sessão.
O resultado vai para results/harness-selftest/<modelo>.json.
"""

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from new_run import EVAL, PYTHON  # noqa: E402
from run_opencode import opencode_config, summarize  # noqa: E402

PROMPT = """Do exactly these steps, one tool call each, and then reply DONE:
1. Run the bash command: pwd
2. Use the read tool on {outside}
   (it may be denied; that is expected, just continue).
3. Run the bash command: {python} -c "import torchxrayvision; print('txv', torchxrayvision.__version__)"
4. Use the task tool with subagent_type "analyst" and ask it to create the file {sub} containing the word ok."""


def export(sid: str, target: pathlib.Path, cwd: pathlib.Path) -> dict:
    with open(target, "w") as f:
        subprocess.run(["opencode", "export", sid], stdout=f, stderr=subprocess.DEVNULL, cwd=cwd)
    return json.loads(target.read_text())


def selftest(model: str, provider: str) -> dict:
    run_dir = pathlib.Path(tempfile.mkdtemp(prefix=f"selftest-{model}-", dir=os.environ["TMPDIR"]))
    (run_dir / "opencode.json").write_text(json.dumps(opencode_config(provider, model, run_dir), indent=2))
    outside = EVAL / "README.md"
    sub = run_dir / "sub.txt"
    events = run_dir / "events.jsonl"
    with open(events, "w") as out:
        subprocess.run(
            ["opencode", "run", "--model", f"{provider}/{model}", "--format", "json", "--auto", "--dir", str(run_dir),
             PROMPT.format(outside=outside, python=PYTHON, sub=sub)],
            cwd=run_dir, env={**os.environ, "PWD": str(run_dir)}, stdout=out, stderr=subprocess.DEVNULL, timeout=900,
        )
    summary = summarize(events)
    main = export(summary["main_session"], run_dir / "main.json", run_dir)
    children = [export(s, run_dir / f"{s}.json", run_dir) for s in summary["child_sessions"]]

    tool_parts = [p for m in main["messages"] for p in m["parts"] if p.get("type") == "tool"]
    reads_outside = [p for p in tool_parts if p["tool"] == "read" and str(outside) in json.dumps(p["state"].get("input", {}))]
    bash_out = " ".join(str(p["state"].get("output", "")) for p in tool_parts if p["tool"] == "bash")
    result = {
        "model": model,
        "session_directory_is_run_dir": main["info"].get("directory") == str(run_dir),
        "session_directory": main["info"].get("directory"),
        "outside_read_attempted": bool(reads_outside),
        "outside_read_denied": bool(reads_outside) and all(p["state"].get("status") == "error" for p in reads_outside),
        "project_python_works": "txv " in bash_out,
        "subagents": [{"agent": c["info"].get("agent"), "model": (c["info"].get("model") or {}).get("id")} for c in children],
        "subagent_file_written": sub.exists() and "ok" in sub.read_text().lower(),
    }
    result["subagent_same_model"] = bool(result["subagents"]) and all(s["model"] == model for s in result["subagents"])
    result["passed"] = all(result[k] for k in ["session_directory_is_run_dir", "outside_read_denied", "project_python_works",
                                                "subagent_same_model", "subagent_file_written"])
    shutil.rmtree(run_dir)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--provider", default="DGX-UFSC")
    args = parser.parse_args()

    out_dir = EVAL / "results" / "harness-selftest"
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = True
    for model in args.model:
        result = selftest(model, args.provider)
        (out_dir / f"{model}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(result, ensure_ascii=False))
        ok &= result["passed"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
