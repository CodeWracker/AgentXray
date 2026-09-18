"""Confere se uma rodada cumpriu o contrato e se os resultados são reprodutíveis.

Uso:
    tools/py tools/check_run.py results/v2/single-agent/qwen3.8-27b/20260918-101500-ID001
    tools/py tools/check_run.py results/v1/single-agent/qwen3.8-27b      # rodadas v1: so o JSON

Para cada `*_analysis/` da rodada:
  1. o JSON final existe, é válido, tem exatamente os quatro campos e o nome da imagem certo;
  2. os arquivos exigidos pelo modo (single/agents) existem;
  3. `scripts/reproduce.py` roda numa cópia limpa e recria, byte a byte, tudo que está em
     `images/` (menos o original) e em `measurements/`.

O relatório vai para `check_report.json` dentro da rodada (exceto em rodadas sem manifesto, que são
tratadas como legado e só têm o JSON verificado).
"""

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

EVAL = pathlib.Path(__file__).resolve().parent.parent
JSON_KEYS = {"image", "findings", "impression", "limitations"}
ALLOWED_KEYS = {"image", "findings", "impression", "limitations", "differential_diagnosis", "localized_lesions"}
# contrato da v2; a v1 só exige o que o próprio prompt v1 pedia e não tem reprodução
REQUIRED_V1 = {
    "single": ["analyze_image.py"],
    "agents": ["planning/processing_plan.txt"],
}
REQUIRED = {
    "single": ["provenance/first_look.md", "provenance/decisions.md", "provenance/tools.json", "scripts/reproduce.py"],
    "agents": [
        "planning/processing_plan.txt",
        "planning/final_synthesis.txt",
        "provenance/tools.json",
        "scripts/reproduce.py",
    ],
}


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derived_outputs(analysis: pathlib.Path) -> dict[str, str]:
    files = [p for p in (analysis / "images").glob("*") if p.is_file() and not p.name.startswith("00_original.")]
    files += [p for p in (analysis / "measurements").rglob("*") if p.is_file()]
    return {str(p.relative_to(analysis)): sha256(p) for p in sorted(files)}


def check_json(analysis: pathlib.Path, image_name: str) -> list[str]:
    path = analysis / f"{image_name}.json"
    if not path.exists():
        return [f"JSON final ausente: {path.name}"]
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return [f"JSON inválido: {e}"]
    errors = []
    if not isinstance(data, dict) or not JSON_KEYS.issubset(set(data)) or not set(data).issubset(ALLOWED_KEYS):
        errors.append(f"campos do JSON diferentes do esperado: {sorted(data) if isinstance(data, dict) else type(data).__name__}")
    elif data["image"] != image_name:
        errors.append(f'"image" = {data["image"]!r}, esperado {image_name!r}')
    return errors


def check_tools(analysis: pathlib.Path) -> list[str]:
    path = analysis / "provenance" / "tools.json"
    if not path.exists():
        return []
    try:
        tools = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return [f"provenance/tools.json inválido: {e}"]
    if not isinstance(tools, list) or not all(isinstance(t, dict) and {"tool", "used_for", "trusted"} <= set(t) for t in tools):
        return ["provenance/tools.json deve ser uma lista de objetos com tool, used_for e trusted"]
    return []


def check_reproduce(analysis: pathlib.Path, timeout: int) -> dict:
    expected = derived_outputs(analysis)
    with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as tmp:
        clone = pathlib.Path(tmp) / analysis.name
        shutil.copytree(analysis, clone)
        for rel in expected:
            (clone / rel).unlink()
        try:
            proc = subprocess.run(
                [str(EVAL / "tools" / "py"), "scripts/reproduce.py"],
                cwd=clone, capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"timeout de {timeout}s"}
        if proc.returncode != 0:
            return {"ok": False, "error": "reproduce.py falhou", "stderr": proc.stderr[-2000:]}
        produced = derived_outputs(clone)

    missing = sorted(set(expected) - set(produced))
    extra = sorted(set(produced) - set(expected))
    different = sorted(k for k in set(expected) & set(produced) if expected[k] != produced[k])
    return {
        "ok": not (missing or extra or different),
        "outputs": len(expected),
        "missing": missing,
        "extra": extra,
        "different": different,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=pathlib.Path)
    parser.add_argument("--timeout", type=int, default=900, help="segundos por reproduce.py")
    parser.add_argument("--no-reproduce", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    mode = manifest["mode"] if manifest else None
    required = (REQUIRED_V1 if manifest and manifest.get("prompt_version") == "v1" else REQUIRED) if manifest else {}

    report = {}
    for analysis in sorted(run_dir.rglob("*_analysis")):
        if not analysis.is_dir():
            continue
        stem = analysis.name.removesuffix("_analysis")
        candidates = [p.name for p in analysis.parent.glob(f"{stem}.*") if p.is_file()]
        image_name = candidates[0] if candidates else f"{stem}.png"

        entry = {"errors": check_json(analysis, image_name)}
        if mode:
            entry["errors"] += [f"ausente: {rel}" for rel in required[mode] if not (analysis / rel).exists()]
            entry["errors"] += check_tools(analysis)
            if not args.no_reproduce and (analysis / "scripts" / "reproduce.py").exists():
                entry["reproduce"] = check_reproduce(analysis, args.timeout)
        report[str(analysis.relative_to(run_dir))] = entry

        status = "ok" if not entry["errors"] and entry.get("reproduce", {"ok": True})["ok"] else "FALHA"
        print(f"[{status}] {analysis.relative_to(run_dir)}")
        for error in entry["errors"]:
            print(f"    - {error}")
        if "reproduce" in entry and not entry["reproduce"]["ok"]:
            print(f"    - reprodução: {json.dumps(entry['reproduce'], ensure_ascii=False)}")

    if not report:
        print("nenhuma pasta *_analysis encontrada", file=sys.stderr)
        return 1
    if manifest:
        (run_dir / "check_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    failed = any(e["errors"] or not e.get("reproduce", {"ok": True})["ok"] for e in report.values())
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
