#!/usr/bin/env python3
"""Analises da escada de comparacao que complementam exp1_metrics.py, exp2_metrics.py e collect.py.

Gera, em analysis/ladder/:
  classifier_agreement.csv  o modelo concorda com o classificador no topo? quando discorda, quem acerta?
  consistency.csv           repeticoes r1 a r3 do agente unico aberto no Exp 1
  cost.csv                  tokens e tempo por caso em cada condicao (zero-shot pelos _usage, agentes pelo coletor)
  compliance.csv            protocolo do harness nas condicoes com agente
  applicability.csv         uso e confianca declarada em modelos que servem ou nao para a imagem
  anatomy.csv               a regiao do corpo foi reconhecida sem ter sido informada?
Linhas de condicoes que ainda nao terminaram saem com status in_progress; as figuras e tabelas marcam isso.

Uso (depois de exp1_metrics.py, exp2_metrics.py e collect.py):
    tools/py analysis/ladder_analysis.py
"""

import csv
import json
import pathlib
import re
import statistics
import sys

EVAL = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EVAL / "analysis"))
from align_diagnosis import score_case  # noqa: E402
from exp1_metrics import HINT_CLASSIFIERS, MODELS, collect_agent, load_json  # noqa: E402

RESULTS = EVAL / "results"
ANALYSIS = EVAL / "analysis"
OUT = ANALYSIS / "ladder"
SUPERVISED = RESULTS / "supervised" / "exp1_image_level"
VOCABS = [("open", "v4"), ("closed", "v4c")]
BENCHES = [("exp1", "exp1_image_level"), ("exp2", "exp2_bbox_localization")]
AGENT_MODES = [("single", "single-agent"), ("council", "multi-agent-single-model")]
CHEST = re.compile(r"\b(chest|thorax|thoracic|lungs?|pulmonary|ribs?|cxr|mediastin\w*|hemithorax)\b", re.I)


def read_csv(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(name: str, rows: list[dict]):
    OUT.mkdir(parents=True, exist_ok=True)
    if not rows:
        (OUT / name).write_text("")
        return
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def med(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else ""


def share(flags):
    flags = [f for f in flags if f is not None]
    return sum(flags) / len(flags) if flags else ""


def status_of(summary: dict, cid: str) -> str:
    row = summary.get(cid)
    return row["status"] if row else "not_started"


def agent_runs() -> dict:
    """Uma linha do coletor por (versao, bench, modo, modelo, imagem): a rodada r1 valida mais recente."""
    chosen = {}
    for r in read_csv(ANALYSIS / "summary.csv"):
        parts = pathlib.Path(r["run_dir"]).parts
        if len(parts) < 5 or parts[0] not in ("v4", "v4c") or not parts[4].endswith("-r1"):
            continue
        key = (parts[0], parts[1], parts[2], parts[3], r["image"])
        prev = chosen.get(key)
        valid = r["has_valid_json"].lower() in ("true", "1")
        if prev is None or valid or not prev[1]:
            if prev is None or r["run_dir"] > prev[0]["run_dir"] or (valid and not prev[1]):
                chosen[key] = (r, valid)
    return {k: v[0] for k, v in chosen.items()}


def classifier_agreement(summary1: dict) -> list[dict]:
    cases = read_csv(ANALYSIS / "exp1" / "cases.csv")
    top1 = {}
    for h in read_csv(ANALYSIS / "exp1" / "hypotheses.csv"):
        if h["position"] == "1":
            top1[(h["condition"], h["image"])] = set(filter(None, h["classes"].split("|")))
    by_cond = {}
    for c in cases:
        by_cond.setdefault(c["condition"], []).append(c)
    preds = {w: (load_json(SUPERVISED / w / "predictions.json") or {}).get("predictions", {})
             for w in HINT_CLASSIFIERS.values()}
    runs = agent_runs()
    rows = []
    for vocab, version in VOCABS:
        suffix = "" if vocab == "open" else ":closed"
        for model in MODELS:
            for mode in [*HINT_CLASSIFIERS, "single", "council"]:
                cid = f"{model}:{mode}{suffix}"
                weights = HINT_CLASSIFIERS.get(mode, HINT_CLASSIFIERS["clf_all"])
                row = {"condition": cid, "model": model, "mode": mode, "vocab": vocab, "classifier": weights,
                       "status": status_of(summary1, cid)}
                valid = [c for c in by_cond.get(cid, []) if c["valid"] == "1"]
                counts = dict(n_valid=len(valid), agree=0, agree_correct=0, disagree=0, model_only=0, clf_only=0,
                              both_right=0, both_wrong=0)
                for c in valid:
                    p = preds[weights].get(c["image"])
                    if not p:
                        continue
                    clf_top = p["ranked_classes"][0]
                    gt = set(c["gt"].split("|"))
                    m_top = top1.get((cid, c["image"]), set())
                    m_ok, c_ok = c["hit1"] == "1", clf_top in gt
                    if clf_top in m_top:
                        counts["agree"] += 1
                        counts["agree_correct"] += m_ok
                    else:
                        counts["disagree"] += 1
                        key = ("both_right" if m_ok and c_ok else "model_only" if m_ok else
                               "clf_only" if c_ok else "both_wrong")
                        counts[key] += 1
                if mode in ("single", "council"):
                    mode_dir = dict(AGENT_MODES)[mode]
                    used = [runs[k]["used_torchxrayvision"].lower() in ("true", "1")
                            for k in runs if k[:4] == (version, "exp1_image_level", mode_dir, model)]
                    row["used_classifier_share"] = share(used)
                else:
                    row["used_classifier_share"] = 1.0
                rows.append({**row, **counts})
    return rows


def consistency() -> list[dict]:
    manifest = json.loads((EVAL / "inputs" / "benchmarks" / "exp1_image_level" / "manifest.json").read_text())
    images = [it["image"] for it in manifest]
    gt = {it["image"]: it["finding_labels"] or ["No Finding"] for it in manifest}
    rows = []
    for model in MODELS:
        reps = {rep: collect_agent("single-agent", model, images, rep=rep) for rep in ("r1", "r2", "r3")}
        finished = min(sum(c["finished"] for c in reps[r].values()) for r in reps)
        hits = {r: [] for r in reps}
        tops = {img: [] for img in images}
        for rep, cases in reps.items():
            for img in images:
                s = score_case(cases[img]["hypotheses"], gt[img])
                hits[rep].append(float(s["hit1"]))
                win = s["window"]
                tops[img].append(frozenset(win[0]["classes"]) if win and cases[img]["valid"] else None)
        complete = [t for t in tops.values() if all(x is not None for x in t)]
        same_all = [len(set(t)) == 1 and bool(t[0]) for t in complete]
        pair = [sum(a == b and bool(a) for a, b in [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]) / 3 for t in complete]
        hit_means = [statistics.mean(v) for v in hits.values()]
        rows.append({"model": model, "status": "complete" if finished == len(images) else "in_progress",
                     "finished_min_rep": finished, "cases_all_valid": len(complete),
                     "top1_same_all_three": share(same_all), "top1_pairwise_agreement": share(pair),
                     **{f"hit1_{r}": statistics.mean(hits[r]) for r in hits},
                     "hit1_sd_across_reps": statistics.pstdev(hit_means)})
    return rows


def cost(summary1: dict, summary2: dict) -> list[dict]:
    runs = agent_runs()
    rows = []
    for vocab, version in VOCABS:
        suffix = "" if vocab == "open" else ":closed"
        for short, bench in BENCHES:
            summ = summary1 if short == "exp1" else summary2
            zs_dirs = [("zeroshot", "zeroshot")] + ([(m, f"zeroshot_clf-{w}") for m, w in HINT_CLASSIFIERS.items()]
                                                   if short == "exp1" else [])
            for model in MODELS:
                for mode, zs_dir in zs_dirs:
                    usage = [load_json(f) or {} for f in (RESULTS / version / zs_dir / bench / model / "_usage").glob("*.json")]
                    u = [x.get("usage", {}) for x in usage]
                    rows.append({"experiment": short, "model": model, "mode": mode, "vocab": vocab,
                                 "status": status_of(summ, f"{model}:{mode}{suffix}"), "n": len(u),
                                 "median_tokens_in": med([num(x.get("prompt_tokens")) for x in u]),
                                 "median_tokens_out": med([num(x.get("completion_tokens")) for x in u]),
                                 "median_reasoning": med([num((x.get("completion_tokens_details") or {}).get("reasoning_tokens")) for x in u]),
                                 "median_wall_s": med([num(x.get("elapsed_s")) for x in usage]),
                                 "median_tool_calls": 0,
                                 "truncated_share": share([x.get("finish_reason") == "length" for x in u])})
                for mode, mode_dir in AGENT_MODES:
                    sel = [r for k, r in runs.items() if k[:4] == (version, bench, mode_dir, model)]
                    rows.append({"experiment": short, "model": model, "mode": mode, "vocab": vocab,
                                 "status": status_of(summ, f"{model}:{mode}{suffix}"), "n": len(sel),
                                 "median_tokens_in": med([num(r["tokens_input"]) for r in sel]),
                                 "median_tokens_out": med([num(r["tokens_output"]) for r in sel]),
                                 "median_reasoning": med([num(r["tokens_reasoning"]) for r in sel]),
                                 "median_wall_s": med([num(r["elapsed_s"]) for r in sel]),
                                 "median_tool_calls": med([num(r["tool_calls_total"]) for r in sel]),
                                 "truncated_share": ""})
    return rows


def agent_table(summary1: dict, summary2: dict, fields) -> list[dict]:
    runs = agent_runs()
    rows = []
    for vocab, version in VOCABS:
        suffix = "" if vocab == "open" else ":closed"
        for short, bench in BENCHES:
            summ = summary1 if short == "exp1" else summary2
            for model in MODELS:
                for mode, mode_dir in AGENT_MODES:
                    sel = [r for k, r in runs.items() if k[:4] == (version, bench, mode_dir, model)]
                    row = {"experiment": short, "model": model, "mode": mode, "vocab": vocab,
                           "status": status_of(summ, f"{model}:{mode}{suffix}"), "n": len(sel)}
                    row.update({name: fn(sel) for name, fn in fields})
                    rows.append(row)
    return rows


def flag(col):
    return lambda sel: share([r[col].lower() in ("true", "1") if r[col] != "" else None for r in sel])


def median_of(col):
    return lambda sel: med([num(r[col]) for r in sel])


def anatomy(summary1: dict, summary2: dict) -> list[dict]:
    """Regiao reconhecida: first_look.md do agente, ou findings e impression do zero-shot."""
    rows = []
    runs = agent_runs()
    for vocab, version in VOCABS:
        suffix = "" if vocab == "open" else ":closed"
        for short, bench in BENCHES:
            summ = summary1 if short == "exp1" else summary2
            for model in MODELS:
                files = [load_json(f) for f in (RESULTS / version / "zeroshot" / bench / model).glob("0*.json")]
                texts = [f"{d.get('findings', '')} {d.get('impression', '')}" for d in files if isinstance(d, dict)]
                rows.append({"experiment": short, "model": model, "mode": "zeroshot", "vocab": vocab,
                             "status": status_of(summ, f"{model}:zeroshot{suffix}"), "n": len(texts),
                             "chest_recognized": share([bool(CHEST.search(t)) for t in texts])})
                for mode, mode_dir in AGENT_MODES:
                    sel = [r for k, r in runs.items() if k[:4] == (version, bench, mode_dir, model)]
                    flags = []
                    for r in sel:
                        first = next((RESULTS / r["run_dir"]).glob("*_analysis/provenance/first_look.md"), None)
                        flags.append(bool(CHEST.search(first.read_text(errors="ignore")[:4000])) if first else None)
                    rows.append({"experiment": short, "model": model, "mode": mode, "vocab": vocab,
                                 "status": status_of(summ, f"{model}:{mode}{suffix}"), "n": len(sel),
                                 "chest_recognized": share(flags)})
    return rows


def main() -> int:
    summary1 = {r["condition"]: r for r in read_csv(ANALYSIS / "exp1" / "summary.csv")}
    summary2 = {r["condition"]: r for r in read_csv(ANALYSIS / "exp2" / "summary.csv")}
    write_csv("classifier_agreement.csv", classifier_agreement(summary1))
    write_csv("consistency.csv", consistency())
    write_csv("cost.csv", cost(summary1, summary2))
    write_csv("compliance.csv", agent_table(summary1, summary2, [
        ("valid_json", flag("has_valid_json")), ("check_passed", flag("check_passed")),
        ("reproduce_ok", flag("reproduce_ok")), ("timed_out", flag("timed_out")),
        ("median_nudges", median_of("nudges")), ("median_log_action_ratio", median_of("log_action_ratio")),
        ("median_blocked", median_of("log_blocked_calls")), ("median_path_blocked", median_of("path_blocked_calls")),
        ("tools_declared", lambda sel: share([num(r["declared_tools_count"]) > 0 if num(r["declared_tools_count"]) is not None else None for r in sel])),
    ]))
    write_csv("applicability.csv", agent_table(summary1, summary2, [
        ("used_torchxrayvision", flag("used_torchxrayvision")), ("trusted_chest_model", flag("trusted_chest_model")),
        ("used_inapplicable", lambda sel: share([(r["used_yolo"].lower() in ("true", "1")) or
                                                 (r["used_fracture_classifier"].lower() in ("true", "1")) for r in sel])),
        ("trusted_inapplicable", flag("trusted_fracture_classifier")),
        ("used_biomedclip", flag("used_biomedclip")), ("used_medsam", flag("used_medsam")),
        ("used_cv_classic", flag("used_cv_classic")),
    ]))
    write_csv("anatomy.csv", anatomy(summary1, summary2))
    print(f"analises da escada em {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
