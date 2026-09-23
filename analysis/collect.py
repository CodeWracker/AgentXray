#!/usr/bin/env python3
"""Coleta metricas de processo, inspecao, ferramentas e conteudo de todas as rodadas.

Percorre results/ (v1, v1-runner, v2, v2-pilot, v2-ablation), extrai dados dos
manifestos, harness_result.json, sessões, scripts, provenance/tools.json e JSON final,
e gera analysis/summary.csv com uma linha por rodada e imagem.

Uso:
    tools/py analysis/collect.py [--out analysis/summary.csv]
"""

import argparse
import csv
import json
import os
import pathlib
import re
import sys

EVAL = pathlib.Path(__file__).resolve().parent.parent


def static_analyze_scripts(scripts_dir: pathlib.Path) -> dict:
    loc = 0
    num_scripts = 0
    tools_used = {
        "cv_classic": False,
        "torchxrayvision": False,
        "biomedclip": False,
        "medsam": False,
        "yolo": False,
        "fracture_classifier": False,
    }
    if not scripts_dir.exists():
        return {"loc": 0, "num_scripts": 0, **tools_used}

    py_files = list(scripts_dir.glob("*.py"))
    num_scripts = len(py_files)
    for py in py_files:
        try:
            content = py.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        lines = [line.strip() for line in content.splitlines()]
        code_lines = [l for l in lines if l and not l.startswith("#")]
        loc += len(code_lines)
        low = content.lower()

        if any(k in low for k in ["cv2", "clahe", "canny", "gaussianblur", "skimage", "pil", "pillow", "scipy.ndimage"]):
            tools_used["cv_classic"] = True
        if any(k in low for k in ["torchxrayvision", "xrv"]):
            tools_used["torchxrayvision"] = True
        if any(k in low for k in ["biomedclip", "open_clip"]):
            tools_used["biomedclip"] = True
        if "medsam" in low:
            tools_used["medsam"] = True
        if any(k in low for k in ["grazpedwri", "yolo", "ultralytics"]):
            tools_used["yolo"] = True
        if any(k in low for k in ["bone-fracture-detection", "siglip"]):
            tools_used["fracture_classifier"] = True

    return {"loc": loc, "num_scripts": num_scripts, **tools_used}


def parse_events(events_path: pathlib.Path, run_dir: pathlib.Path, image_name: str) -> dict:
    if not events_path.exists():
        return {
            "first_action_was_read_original": None,
            "images_read_count": 0,
            "read_original": False,
        }

    stem = image_name.split(".")[0]
    read_images = set()
    read_original = False
    first_action_was_read_original = None

    for line in events_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("type") != "tool_use":
            continue
        part = ev.get("part", {})
        tool = part.get("tool")
        inp = part.get("state", {}).get("input", {})

        if tool == "read":
            file_path = str(inp.get("filePath", ""))
            if image_name in file_path or f"{stem}." in file_path or "00_original.png" in file_path:
                read_original = True
                if first_action_was_read_original is None:
                    first_action_was_read_original = True
            for ext in [".png", ".jpg", ".jpeg"]:
                if file_path.lower().endswith(ext):
                    read_images.add(pathlib.Path(file_path).name)
        elif tool in ["bash", "write", "edit"]:
            if first_action_was_read_original is None:
                first_action_was_read_original = False

    return {
        "first_action_was_read_original": bool(first_action_was_read_original),
        "images_read_count": len(read_images),
        "read_original": read_original,
    }


def session_totals(sessions_dir: pathlib.Path) -> dict | None:
    """Soma tokens e chamadas de ferramenta de todas as sessoes exportadas (principal e subagentes).

    O harness_result.json so conta a sessao principal; no conselho quase todo o trabalho fica nos subagentes.
    """
    files = sorted(sessions_dir.glob("*.json")) if sessions_dir.exists() else []
    if not files:
        return None
    tokens = {"input": 0, "output": 0, "reasoning": 0}
    tools: dict[str, int] = {}
    images_read: set[str] = set()
    for f in files:
        try:
            session = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        for msg in session.get("messages", []):
            t = msg.get("info", {}).get("tokens") or {}
            for k in tokens:
                tokens[k] += t.get(k, 0) or 0
            for part in msg.get("parts", []):
                if part.get("type") != "tool":
                    continue
                name = part.get("tool")
                tools[name] = tools.get(name, 0) + 1
                if name == "read":
                    path = str(part.get("state", {}).get("input", {}).get("filePath", ""))
                    if path.lower().endswith((".png", ".jpg", ".jpeg")):
                        images_read.add(pathlib.Path(path).name)
    return {"tokens": tokens, "tools": tools, "images_read": images_read, "sessions": len(files)}


def parse_provenance_tools(tools_path: pathlib.Path) -> dict:
    if not tools_path.exists():
        return {
            "declared_tools_count": 0,
            "trusted_count": 0,
            "untrusted_count": 0,
            "trusted_chest_model": False,
            "trusted_fracture_classifier": False,
        }
    try:
        data = json.loads(tools_path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "declared_tools_count": 0,
            "trusted_count": 0,
            "untrusted_count": 0,
            "trusted_chest_model": False,
            "trusted_fracture_classifier": False,
        }

    if not isinstance(data, list):
        return {
            "declared_tools_count": 0,
            "trusted_count": 0,
            "untrusted_count": 0,
            "trusted_chest_model": False,
            "trusted_fracture_classifier": False,
        }

    declared = len(data)
    trusted = 0
    untrusted = 0
    trusted_chest = False
    trusted_fracture = False

    for item in data:
        if not isinstance(item, dict):
            continue
        is_trusted = item.get("trusted")
        tool_name = str(item.get("tool", "")).lower()
        if is_trusted is True:
            trusted += 1
            if any(k in tool_name for k in ["densenet", "torchxrayvision", "xrv", "chest"]):
                trusted_chest = True
            if any(k in tool_name for k in ["bone-fracture", "siglip"]):
                trusted_fracture = True
        elif is_trusted is False:
            untrusted += 1

    return {
        "declared_tools_count": declared,
        "trusted_count": trusted,
        "untrusted_count": untrusted,
        "trusted_chest_model": trusted_chest,
        "trusted_fracture_classifier": trusted_fracture,
    }


def collect_one(analysis_dir: pathlib.Path, results_root: pathlib.Path) -> dict | None:
    rel_parts = analysis_dir.relative_to(results_root).parts
    if not rel_parts:
        return None

    condition = rel_parts[0]
    is_v1_legacy = condition == "v1"

    run_dir = analysis_dir.parent
    manifest_path = run_dir / "run_manifest.json"
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

    harness_path = run_dir / "harness_result.json"
    harness = {}
    if harness_path.exists():
        try:
            harness = json.loads(harness_path.read_text(encoding="utf-8"))
        except Exception:
            harness = {}

    stem = analysis_dir.name.removesuffix("_analysis")
    image_candidates = [p.name for p in run_dir.glob(f"{stem}.*") if p.is_file() and p.suffix in [".png", ".jpg"]]
    image_name = image_candidates[0] if image_candidates else f"{stem}.png"

    final_json_path = analysis_dir / f"{image_name}.json"
    if not final_json_path.exists():
        alt = list(analysis_dir.glob("*.json"))
        alt_filtered = [p for p in alt if "tools" not in p.name and "report" not in p.name]
        if alt_filtered:
            final_json_path = alt_filtered[0]

    final_json = {}
    if final_json_path.exists():
        try:
            final_json = json.loads(final_json_path.read_text(encoding="utf-8"))
        except Exception:
            final_json = {}

    if manifest:
        mode = manifest.get("mode")
        model = manifest.get("model")
        prompt_version = manifest.get("prompt_version")
        condition_name = manifest.get("condition", condition)
        run_id = run_dir.name
    elif is_v1_legacy:
        mode = rel_parts[1]
        model = rel_parts[2]
        prompt_version = "v1"
        condition_name = "v1"
        run_id = f"v1-{mode}-{model}"
    else:
        mode = rel_parts[1] if len(rel_parts) > 1 else "unknown"
        model = rel_parts[2] if len(rel_parts) > 2 else "unknown"
        prompt_version = "unknown"
        condition_name = condition
        run_id = run_dir.name

    if model in ["sonnet5", "claude"] or "sonnet" in str(model).lower():
        return None

    tokens = harness.get("tokens_main_session", {})
    tool_calls = harness.get("tool_calls", {})

    check_path = run_dir / "check_report.json"
    check_report = {}
    if check_path.exists():
        try:
            check_report = json.loads(check_path.read_text(encoding="utf-8"))
        except Exception:
            check_report = {}

    analysis_key = str(analysis_dir.relative_to(run_dir))
    check_entry = check_report.get(analysis_key, {})
    reproduce = check_entry.get("reproduce", {})

    scripts_dir = analysis_dir / "scripts"
    if not scripts_dir.exists():
        scripts_dir = analysis_dir
    script_stats = static_analyze_scripts(scripts_dir)

    images_dir = analysis_dir / "images"
    gen_images = [p for p in images_dir.glob("*") if p.is_file() and not p.name.startswith("00_original.")] if images_dir.exists() else []

    events_path = run_dir / "opencode_events.jsonl"
    event_stats = parse_events(events_path, run_dir, image_name)

    prov_tools = parse_provenance_tools(analysis_dir / "provenance" / "tools.json")

    has_valid_json = bool(
        isinstance(final_json, dict) and {"image", "findings", "impression", "limitations"} <= set(final_json)
    )

    child_sessions = harness.get("child_sessions", [])
    num_subagents = len(child_sessions)

    # totais de todas as sessoes exportadas; sem exportacao, cai para a sessao principal do harness
    totals = session_totals(run_dir / "sessions")
    tokens_all = totals["tokens"] if totals else tokens
    tools_all = totals["tools"] if totals else tool_calls
    images_read_all = len(totals["images_read"]) if totals else event_stats["images_read_count"]
    gen_names = {p.name for p in gen_images}
    viewed_generated = len(gen_names & totals["images_read"]) if totals else None

    # sem imagem gerada a cobertura nao existe (nao e zero)
    coverage = viewed_generated / len(gen_names) if (viewed_generated is not None and gen_names) else None
    parts = pathlib.Path(str(run_dir.relative_to(results_root))).parts
    experiment = next((p for p in parts if p.startswith("exp")), "")

    return {
        "condition": condition_name,
        "experiment": experiment,
        "prompt_version": prompt_version,
        "mode": mode,
        "model": model,
        "run_id": run_id,
        "image": image_name,
        "run_dir": str(run_dir.relative_to(results_root)),
        "exit_code": harness.get("exit_code"),
        "timed_out": harness.get("timed_out", False),
        "elapsed_s": harness.get("elapsed_s"),
        "nudges": harness.get("nudges", 0),
        "final_json_written": final_json_path.exists(),
        "has_valid_json": has_valid_json,
        "check_passed": harness.get("check_passed"),
        "tokens_input": tokens_all.get("input", 0),
        "tokens_output": tokens_all.get("output", 0),
        "tokens_reasoning": tokens_all.get("reasoning", 0),
        "tokens_input_main": tokens.get("input", 0),
        "tokens_output_main": tokens.get("output", 0),
        "tool_calls_total": sum(tools_all.values()) if tools_all else 0,
        "tool_calls_bash": tools_all.get("bash", 0),
        "tool_calls_read": tools_all.get("read", 0),
        "tool_calls_write": tools_all.get("write", 0),
        "tool_calls_edit": tools_all.get("edit", 0),
        "tool_calls_task": tools_all.get("task", 0),
        "sessions_exported": totals["sessions"] if totals else 0,
        "subagents_count": num_subagents,
        "subagents_ge_6": num_subagents >= 6,
        "images_generated_count": len(gen_images),
        "images_read_count": images_read_all,
        "inspection_coverage": "" if coverage is None else round(coverage, 3),
        "original_inspected_first": event_stats["first_action_was_read_original"],
        "script_loc": script_stats["loc"],
        "num_scripts": script_stats["num_scripts"],
        "used_cv_classic": script_stats["cv_classic"],
        "used_torchxrayvision": script_stats["torchxrayvision"],
        "used_biomedclip": script_stats["biomedclip"],
        "used_medsam": script_stats["medsam"],
        "used_yolo": script_stats["yolo"],
        "used_fracture_classifier": script_stats["fracture_classifier"],
        "declared_tools_count": prov_tools["declared_tools_count"],
        "trusted_tools_count": prov_tools["trusted_count"],
        "untrusted_tools_count": prov_tools["untrusted_count"],
        "trusted_chest_model": prov_tools["trusted_chest_model"],
        "trusted_fracture_classifier": prov_tools["trusted_fracture_classifier"],
        "reproduce_ok": reproduce.get("ok"),
        "reproduce_outputs": reproduce.get("outputs", 0),
        "findings_chars": len(str(final_json.get("findings", ""))),
        "impression_chars": len(str(final_json.get("impression", ""))),
        "limitations_chars": len(str(final_json.get("limitations", ""))),
        "findings": str(final_json.get("findings", "")),
        "impression": str(final_json.get("impression", "")),
        "limitations": str(final_json.get("limitations", "")),
    }


def main():
    parser = argparse.ArgumentParser(description="Coleta resumo estatistico das rodadas.")
    parser.add_argument("--results", type=pathlib.Path, default=EVAL / "results", help="diretorio results/")
    parser.add_argument("--out", type=pathlib.Path, default=EVAL / "analysis" / "summary.csv", help="arquivo CSV de saida")
    args = parser.parse_args()

    results_root = args.results.resolve()
    analysis_dirs = sorted(results_root.rglob("*_analysis"))

    rows = []
    for ad in analysis_dirs:
        if not ad.is_dir():
            continue
        row = collect_one(ad, results_root)
        if row:
            rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print("Nenhuma rodada encontrada em", results_root, file=sys.stderr)
        return 1

    fieldnames = list(rows[0].keys())
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Coletadas {len(rows)} linhas em {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
