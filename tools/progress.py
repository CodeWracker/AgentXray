#!/usr/bin/env python3
"""Monitora o progresso das rodadas v2 em execucao.

Uso:
    tools/py tools/progress.py
"""

import json
import pathlib
import sys

EVAL = pathlib.Path(__file__).resolve().parent.parent
RESULTS_V2 = EVAL / "results" / "v2"

MODELS = ["qwen3.8-27b", "qwen3.6-27b", "gemma4-26b"]
MODES = ["single-agent", "multi-agent-single-model"]
IMAGES = ["ID001-xray.png", "ID002-xray.png", "ID003-xray.png"]
TOTAL_EXPECTED = len(MODELS) * len(MODES) * len(IMAGES)  # 3 * 2 * 3 = 18


def check_progress() -> dict:
    completed = []
    in_progress = []
    failed = []

    if not RESULTS_V2.exists():
        return {
            "total_expected": TOTAL_EXPECTED,
            "completed_count": 0,
            "in_progress_count": 0,
            "failed_count": 0,
            "percentage": 0.0,
            "completed": [],
            "in_progress": [],
            "failed": [],
        }

    for run_dir in RESULTS_V2.glob("**/harness_result.json"):
        parent = run_dir.parent
        manifest_file = parent / "run_manifest.json"
        events_file = parent / "opencode_events.jsonl"
        harness_file = run_dir

        mdata = {}
        if manifest_file.exists():
            try:
                mdata = json.loads(manifest_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        hdata = {}
        try:
            hdata = json.loads(harness_file.read_text(encoding="utf-8"))
        except Exception:
            pass

        imgs = mdata.get("images", [])
        image_name = imgs[0].get("file", parent.name) if imgs else parent.name

        info = {
            "model": mdata.get("model", "unknown"),
            "mode": mdata.get("mode", "unknown"),
            "condition": mdata.get("condition", parent.parent.parent.name),
            "image": image_name,
            "run_dir": parent.name,
            "elapsed_s": hdata.get("elapsed_s", 0),
            "nudges": hdata.get("nudges", 0),
            "final_json_written": hdata.get("final_json_written", False),
            "check_passed": hdata.get("check_passed", False),
        }

        if info["final_json_written"]:
            completed.append(info)
        else:
            failed.append(info)

    # Verifica sessoes ativas que ainda nao tem harness_result.json
    for ev_file in RESULTS_V2.glob("**/opencode_events.jsonl"):
        parent = ev_file.parent
        if (parent / "harness_result.json").exists():
            continue
        manifest_file = parent / "run_manifest.json"
        mdata = {}
        if manifest_file.exists():
            try:
                mdata = json.loads(manifest_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        imgs = mdata.get("images", [])
        image_name = imgs[0].get("file", parent.name) if imgs else parent.name

        lines = len(ev_file.read_text(encoding="utf-8", errors="ignore").splitlines())
        in_progress.append({
            "model": mdata.get("model", "unknown"),
            "mode": mdata.get("mode", "unknown"),
            "condition": mdata.get("condition", parent.parent.parent.name),
            "image": image_name,
            "run_dir": parent.name,
            "events_lines": lines,
        })

    # Benchmark total target: 45 (exp1) + 48 (exp2) = 93 per model * 3 = 279
    TOTAL_TARGET = 93 * len(MODELS)
    pct = (len(completed) / max(1, TOTAL_TARGET)) * 100.0
    return {
        "total_target": TOTAL_TARGET,
        "completed_count": len(completed),
        "in_progress_count": len(in_progress),
        "failed_count": len(failed),
        "percentage": round(pct, 1),
        "completed": completed,
        "in_progress": in_progress,
        "failed": failed,
    }


def main():
    prog = check_progress()
    print(f"Progresso Geral: {prog['percentage']:.1f}% ({prog['completed_count']}/{prog['total_target']} sessoes concluidas)")
    print(f"Em andamento: {prog['in_progress_count']} sessoes | Falhas: {prog['failed_count']}")

    if prog["completed"]:
        print("\nSessoes Concluidas:")
        for c in prog["completed"]:
            status = "check_ok" if c.get("check_passed") else "check_falha"
            print(f"  [OK] {c['model']} | {c['mode']} | {c['image']} ({c.get('elapsed_s', 0):.1f}s, nudges={c.get('nudges', 0)}, {status})")

    if prog["in_progress"]:
        print("\nSessoes em Andamento:")
        for p in prog["in_progress"]:
            print(f"  [RUNNING] {p['model']} | {p['mode']} | {p['image']} (linhas de evento: {p.get('events_lines', 0)})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
