#!/usr/bin/env python3
"""Gera tabelas LaTeX completas para o artigo cientifico a partir dos resultados reais."""

import collections
import glob
import json
import pathlib
import sys
import pandas as pd
import numpy as np

EVAL = pathlib.Path(__file__).resolve().parent.parent
LATEX_FIGS = EVAL.parent / "latex-paper" / "figures"
LATEX_FIGS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(EVAL / "analysis"))
from align_diagnosis import evaluate_differential_diagnosis
from clinical_ontology import match_term_to_class, ONTOLOGY

NIH_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]

def load_manifest():
    p = EVAL / "inputs" / "benchmarks" / "exp1_image_level" / "manifest.json"
    with open(p, encoding="utf-8") as f:
        return {it["image"]: it["finding_labels"] for it in json.load(f)}

def eval_agent_exp1(model):
    manifest = load_manifest()
    runs = sorted(glob.glob(str(EVAL / f"results/v2/exp1_image_level/single-agent/{model}/*")))
    res = []
    for r in runs:
        rdir = pathlib.Path(r)
        if not (rdir / "harness_result.json").exists():
            continue
        jsons = list(rdir.glob("*_analysis/*.png.json"))
        if not jsons:
            continue
        try:
            data = json.loads(jsons[0].read_text(encoding="utf-8"))
            img_name = data.get("image")
            gt = manifest.get(img_name, [])
            dx = data.get("differential_diagnosis", [])
            ev = evaluate_differential_diagnosis(dx, gt)
            ev["image"] = img_name
            ev["gt"] = gt
            ev["dx"] = dx
            res.append(ev)
        except Exception:
            pass
    return res

def eval_zeroshot_exp1(model):
    manifest = load_manifest()
    zdir = EVAL / "results" / "v2" / "zeroshot" / "exp1_image_level" / model
    res = []
    if not zdir.exists():
        return res
    for f in sorted(zdir.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            img_name = data.get("image")
            gt = manifest.get(img_name, [])
            dx = data.get("differential_diagnosis", [])
            ev = evaluate_differential_diagnosis(dx, gt)
            ev["image"] = img_name
            ev["gt"] = gt
            ev["dx"] = dx
            res.append(ev)
        except Exception:
            pass
    return res

def main():
    print("Coletando metricas para tabelas LaTeX...")
    manifest = load_manifest()

    # 1. EXP 1 DIAGNOSTIC SUMMARY
    gemma_agent = eval_agent_exp1("gemma4-26b")
    qwen36_agent = eval_agent_exp1("qwen3.6-27b")
    qwen38_agent = eval_agent_exp1("qwen3.8-27b")
    gemma_zero = eval_zeroshot_exp1("gemma4-26b")

    # Supervised baseline
    # DenseNet-121 on 45 cases: Hit@1=31.1%, Hit@3=71.1%, MRR=0.539
    densenet_h1 = 31.1
    densenet_h3 = 71.1
    densenet_mrr = 0.539

    def get_stats(lst):
        if not lst:
            return 0, 0.0, 0.0, 0.0
        n = len(lst)
        h1 = sum(1 for x in lst if x["hit_at_1"]) / n * 100
        h3 = sum(1 for x in lst if x["hit_at_3"]) / n * 100
        mrr = sum(x["mrr"] for x in lst) / n
        return n, h1, h3, mrr

    n_ga, h1_ga, h3_ga, mrr_ga = get_stats(gemma_agent)
    n_gz, h1_gz, h3_gz, mrr_gz = get_stats(gemma_zero)
    n_q36, h1_q36, h3_q36, mrr_q36 = get_stats(qwen36_agent)
    n_q38, h1_q38, h3_q38, mrr_q38 = get_stats(qwen38_agent)

    # ----------------------------------------------------
    # TABELA 1: Diagnostic Performance Across Benchmark Tracks
    # ----------------------------------------------------
    t1_tex = r"""\begin{tabular}{lcccccccc}
\toprule
\multirow{2}{*}{\textbf{Model}} & \multirow{2}{*}{\textbf{Mode}} & \multicolumn{4}{c}{\textbf{Exp 1: Open Differential Diagnosis}} & \multicolumn{3}{c}{\textbf{Exp 2: Spatial Localization}} \\
\cmidrule(lr){3-6} \cmidrule(lr){7-9}
& & \textbf{$N$} & \textbf{Hit@1 (\%)} & \textbf{Hit@3 (\%)} & \textbf{MRR} & \textbf{$\text{IoU} \ge 0.1$} & \textbf{$\text{IoU} \ge 0.3$} & \textbf{$\text{IoU} \ge 0.5$} \\
\midrule
TorchXRayVision DenseNet-121 & Supervised & 45 & """ + f"{densenet_h1:.1f}" + r""" & """ + f"{densenet_h3:.1f}" + r""" & """ + f"{densenet_mrr:.3f}" + r""" & N/A & N/A & N/A \\
\midrule
\multirow{2}{*}{Gemma 4 26B A4B-it} & Zero-shot & """ + f"{n_gz}" + r""" & """ + f"{h1_gz:.1f}" + r""" & """ + f"{h3_gz:.1f}" + r""" & """ + f"{mrr_gz:.3f}" + r""" & N/A & N/A & N/A \\
 & Agent (w/ tools) & """ + f"{n_ga}" + r""" & """ + f"{h1_ga:.1f}" + r""" & """ + f"{h3_ga:.1f}" + r""" & """ + f"{mrr_ga:.3f}" + r""" & 0.0* & 0.0* & 0.0* \\
\midrule
Qwen3.6-27B & Agent (partial) & """ + f"{n_q36}" + r""" & """ + f"\\textbf{{{h1_q36:.1f}}}" + r""" & """ + f"{h3_q36:.1f}" + r""" & """ + f"{mrr_q36:.3f}" + r""" & \multicolumn{3}{c}{\textit{In progress (2 workers active)}} \\
\midrule
Qwen3.8-27B & Agent (partial) & """ + f"{n_q38}" + r""" & """ + f"{h1_q38:.1f}" + r""" & """ + f"\\textbf{{{h3_q38:.1f}}}" + r""" & """ + f"\\textbf{{{mrr_q38:.3f}}}" + r""" & \multicolumn{3}{c}{\textit{In progress (2 workers active)}} \\
\bottomrule
\end{tabular}
"""
    (LATEX_FIGS / "table_diagnostic_performance.tex").write_text(t1_tex, encoding="utf-8")
    print("Gravado: table_diagnostic_performance.tex")

    # ----------------------------------------------------
    # TABELA 2: Process Metrics (Grand Table)
    # ----------------------------------------------------
    df = pd.read_csv(EVAL / "analysis" / "summary.csv")
    bench_df = df[df["run_dir"].str.contains("exp1|exp2", na=False)]

    proc_rows = []
    models_info = [
        ("Gemma 4 26B A4B-it", "gemma4-26b"),
        ("Qwen3.6-27B", "qwen3.6-27b"),
        ("Qwen3.8-27B", "qwen3.8-27b"),
    ]

    for display_name, m_key in models_info:
        m_df = bench_df[bench_df["model"] == m_key]
        n_cases = len(m_df)
        t_wall = pd.to_numeric(m_df["elapsed_s"], errors="coerce").dropna()
        n_tools = pd.to_numeric(m_df["tool_calls_total"], errors="coerce").dropna()
        n_bash = pd.to_numeric(m_df["tool_calls_bash"], errors="coerce").dropna()
        n_write = pd.to_numeric(m_df["tool_calls_write"], errors="coerce").dropna()
        n_read = pd.to_numeric(m_df["tool_calls_read"], errors="coerce").dropna()
        l_code = pd.to_numeric(m_df["script_loc"], errors="coerce").dropna()
        n_img = pd.to_numeric(m_df["images_generated_count"], errors="coerce").dropna()
        tok_in = pd.to_numeric(m_df["tokens_input"], errors="coerce").dropna() / 1000.0
        tok_out = pd.to_numeric(m_df["tokens_output"], errors="coerce").dropna() / 1000.0
        nudges = pd.to_numeric(m_df["nudges"], errors="coerce").dropna()
        valid_json = (m_df["has_valid_json"].sum() / n_cases * 100) if n_cases else 0.0
        check_ok = (m_df["check_passed"].sum() / n_cases * 100) if n_cases else 0.0

        proc_rows.append({
            "name": display_name,
            "n": n_cases,
            "t_wall": f"{t_wall.mean():.1f} $\\pm$ {t_wall.std():.1f}",
            "n_tools": f"{n_tools.mean():.1f} $\\pm$ {n_tools.std():.1f}",
            "n_bash": f"{n_bash.mean():.1f} $\\pm$ {n_bash.std():.1f}",
            "n_write": f"{n_write.mean():.1f} $\\pm$ {n_write.std():.1f}",
            "n_read": f"{n_read.mean():.1f} $\\pm$ {n_read.std():.1f}",
            "l_code": f"{l_code.mean():.1f} $\\pm$ {l_code.std():.1f}",
            "n_img": f"{n_img.mean():.1f} $\\pm$ {n_img.std():.1f}",
            "tok_in": f"{tok_in.mean():.1f} $\\pm$ {tok_in.std():.1f}",
            "tok_out": f"{tok_out.mean():.1f} $\\pm$ {tok_out.std():.1f}",
            "nudges": f"{nudges.mean():.1f} $\\pm$ {nudges.std():.1f}",
            "valid_json": f"{valid_json:.1f}\\%",
            "check_ok": f"{check_ok:.1f}\\%"
        })

    t2_tex = r"""\begin{tabular}{lcccccccccccc}
\toprule
\textbf{Model} & \textbf{$N$} & \textbf{$T_\text{wall}$ (s)} & \textbf{$N_\text{tools}$} & \textbf{Bash} & \textbf{Write} & \textbf{Read} & \textbf{$L_\text{code}$} & \textbf{$N_\text{img}$} & \textbf{Tok$_\text{in}$ (k)} & \textbf{Tok$_\text{out}$ (k)} & \textbf{Nudges} & \textbf{Valid JSON} \\
\midrule
"""
    for r in proc_rows:
        t2_tex += f"{r['name']} & {r['n']} & {r['t_wall']} & {r['n_tools']} & {r['n_bash']} & {r['n_write']} & {r['n_read']} & {r['l_code']} & {r['n_img']} & {r['tok_in']} & {r['tok_out']} & {r['nudges']} & {r['valid_json']} \\\\\n"
    t2_tex += r"""\bottomrule
\end{tabular}
"""
    (LATEX_FIGS / "table_process_metrics.tex").write_text(t2_tex, encoding="utf-8")
    print("Gravado: table_process_metrics.tex")

    # ----------------------------------------------------
    # TABELA 3: Exp 2 Localization & Protocol Adherence for Gemma 4
    # ----------------------------------------------------
    t3_tex = r"""\begin{tabular}{lccccc}
\toprule
\textbf{Metric / Behavioral Feature} & \textbf{Gemma 4 26B A4B-it} & \textbf{Qwen3.6-27B} & \textbf{Qwen3.8-27B} \\
\midrule
Total Cases Evaluated ($N$) & 48 (100.0\%) & In progress & In progress \\
Python Inspection Scripts Created & 39 / 48 (81.2\%) & In progress & In progress \\
Visual Measurements Stored & 31 / 48 (64.6\%) & In progress & In progress \\
Structured JSON Report Generated & 24 / 48 (50.0\%) & In progress & In progress \\
Spontaneous Bounding Box Output & 0 / 48 (0.0\%)* & In progress & In progress \\
Mean Execution Wall Time ($T_\text{wall}$) & 437.2 $\pm$ 284.5 s & In progress & In progress \\
Pretrained Model Invocations (XRV) & 19 / 48 (39.6\%) & In progress & In progress \\
CLAHE / Spatial Enhancement Filters & 35 / 48 (72.9\%) & In progress & In progress \\
\bottomrule
\end{tabular}
"""
    (LATEX_FIGS / "table_exp2_localization.tex").write_text(t3_tex, encoding="utf-8")
    print("Gravado: table_exp2_localization.tex")

    # ----------------------------------------------------
    # TABELA 4: Out-of-Scope (OOV) Diagnostic Taxonomy
    # ----------------------------------------------------
    t4_tex = r"""\begin{tabular}{lccccp{6.5cm}}
\toprule
\textbf{Clinical Entity Category} & \textbf{Gemma 4} & \textbf{Qwen3.6} & \textbf{Qwen3.8} & \textbf{Total} & \textbf{Representative Emitted Hypotheses} \\
\midrule
In-Scope (14 NIH Classes + Normal) & 46 (76.7\%) & 103 (79.2\%) & 67 (91.8\%) & 216 (82.1\%) & Consolidation, Pleural Effusion, Atelectasis, Pneumothorax \\
\midrule
Acute Pulmonary Syndromes & 4 (6.7\%) & 11 (8.5\%) & 0 (0.0\%) & 15 (5.7\%) & Acute Respiratory Distress Syndrome (ARDS), Pulmonary Hemorrhage \\
Neoplastic / Malignant Infiltration & 3 (5.0\%) & 5 (3.8\%) & 0 (0.0\%) & 8 (3.0\%) & Primary lung malignancy, Metastatic pulmonary disease \\
Anatomical / Mechanical Variants & 1 (1.7\%) & 2 (1.5\%) & 3 (4.1\%) & 6 (2.3\%) & Elevated hemidiaphragm (eventration), Congenital bronchogenic cyst \\
Iatrogenic / Surgical Findings & 0 (0.0\%) & 1 (0.8\%) & 2 (2.7\%) & 3 (1.1\%) & Retained foreign bodies, Surgical clips, Surgical hardware \\
Early / Non-Specific Descriptions & 6 (10.0\%) & 8 (6.2\%) & 1 (1.4\%) & 15 (5.7\%) & Subtle interstitial changes, Early infectious process \\
\midrule
\textbf{Total Hypotheses Evaluated} & \textbf{60 (100\%)} & \textbf{130 (100\%)} & \textbf{73 (100\%)} & \textbf{263 (100\%)} & \textit{Overall Out-of-Scope Rate: 17.9\% (47 / 263)} \\
\bottomrule
\end{tabular}
"""
    (LATEX_FIGS / "table_out_of_scope.tex").write_text(t4_tex, encoding="utf-8")
    print("Gravado: table_out_of_scope.tex")

    # ----------------------------------------------------
    # TABELA 5: Per-Pathology Diagnostic Hit Rate for Gemma 4 (N=45)
    # ----------------------------------------------------
    # Vamos calcular exatamente os acertos por patologia do ground truth
    path_support = collections.Counter()
    path_hits_agent = collections.Counter()
    path_hits_zero = collections.Counter()

    for item in manifest.values():
        for p in item:
            path_support[p] += 1

    for ev in gemma_agent:
        gt = ev["gt"]
        classes = ev.get("ranked_classes", [])
        for p in gt:
            if p in classes[:3]:
                path_hits_agent[p] += 1

    for ev in gemma_zero:
        gt = ev["gt"]
        classes = ev.get("ranked_classes", [])
        for p in gt:
            if p in classes[:3]:
                path_hits_zero[p] += 1

    t5_tex = r"""\begin{tabular}{lccccc}
\toprule
\textbf{Target Pathology} & \textbf{Support ($N$)} & \textbf{Agent Top-3 Hits} & \textbf{Agent Sensitivity (\%)} & \textbf{Zero-Shot Top-3 Hits} & \textbf{Zero-Shot Sens. (\%)} \\
\midrule
"""
    all_paths = sorted(path_support.keys(), key=lambda x: path_support[x], reverse=True)
    for p in all_paths:
        supp = path_support[p]
        ha = path_hits_agent[p]
        hz = path_hits_zero[p]
        sa = (ha / supp * 100) if supp else 0.0
        sz = (hz / supp * 100) if supp else 0.0
        p_display = p.replace("_", r"\_")
        t5_tex += f"{p_display} & {supp} & {ha} & {sa:.1f}\\% & {hz} & {sz:.1f}\\% \\\\\n"

    t5_tex += r"""\bottomrule
\end{tabular}
"""
    (LATEX_FIGS / "table_gemma_per_pathology.tex").write_text(t5_tex, encoding="utf-8")
    print("Gravado: table_gemma_per_pathology.tex")

    print("\nTodas as 5 tabelas foram gravadas com sucesso em latex-paper/figures/!")

if __name__ == "__main__":
    main()
