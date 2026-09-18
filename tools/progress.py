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

    for model in MODELS:
        for mode in MODES:
            mode_dir = RESULTS_V2 / mode / model
            if not mode_dir.exists():
                continue
            for run_dir in sorted(mode_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                harness_json = run_dir / "harness_result.json"
                events_file = run_dir / "opencode_events.jsonl"
                manifest_file = run_dir / "run_manifest.json"

                image_name = "unknown"
                if manifest_file.exists():
                    try:
                        mdata = json.loads(manifest_file.read_text(encoding="utf-8"))
                        imgs = mdata.get("images", [])
                        if imgs:
                            image_name = imgs[0].get("file", "unknown")
                    except Exception:
                        pass

                info = {
                    "model": model,
                    "mode": mode,
                    "image": image_name,
                    "run_dir": run_dir.name,
                }

                if harness_json.exists():
                    try:
                        hdata = json.loads(harness_json.read_text(encoding="utf-8"))
                        info["elapsed_s"] = hdata.get("elapsed_s")
                        info["nudges"] = hdata.get("nudges", 0)
                        info["final_json_written"] = hdata.get("final_json_written", False)
                        info["check_passed"] = hdata.get("check_passed", False)
                        if hdata.get("final_json_written", False):
                            completed.append(info)
                        else:
                            failed.append(info)
                    except Exception:
                        failed.append(info)
                elif events_file.exists():
                    # em andamento
                    info["events_lines"] = len(events_file.read_text(encoding="utf-8", errors="ignore").splitlines())
                    in_progress.append(info)

    pct = (len(completed) / max(1, TOTAL_EXPECTED)) * 100.0
    return {
        "total_expected": TOTAL_EXPECTED,
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
    print(f"Progresso Geral: {prog['percentage']:.1f}% ({prog['completed_count']}/{prog['total_expected']} sessoes concluidas)")
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
