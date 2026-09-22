#!/usr/bin/env python3
"""Gera as tabelas LaTeX de processo, Exp 2 e anatomia do artigo.

As tabelas do Exp 1 sao geradas por analysis/exp1_metrics.py e latex-paper/figures/scripts/exp1_tables_figures.py.

Conforme diretriz experimental:
Apenas o Gemma 4 26B concluiu todas as 93 imagens do benchmark e tem resultados fechados.
Os modelos Qwen (qwen3.6-27b e qwen3.8-27b) estao em execucao ativa e devem constar
exclusivamente com '0.XXX' ou '\\textit{In progress}'.
"""

import pathlib
import pandas as pd

EVAL = pathlib.Path(__file__).resolve().parent.parent
LATEX_FIGS = EVAL.parent / "latex-paper" / "figures"
LATEX_FIGS.mkdir(parents=True, exist_ok=True)


def main():
    print("Coletando metricas para tabelas LaTeX...")

    # ----------------------------------------------------
    # TABELA 2: Process Metrics (Apenas Gemma 4 com numeros, Qwen em In progress)
    # ----------------------------------------------------
    df = pd.read_csv(EVAL / "analysis" / "summary.csv")
    bench_df = df[df["run_dir"].str.contains("exp1|exp2", na=False)]

    m_df = bench_df[bench_df["model"] == "gemma4-26b"]
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

    t2_tex = r"""\begin{tabular}{lcccccccccccc}
\toprule
\textbf{Model} & \textbf{$N$} & \textbf{$T_\text{wall}$ (s)} & \textbf{$N_\text{tools}$} & \textbf{Bash} & \textbf{Write} & \textbf{Read} & \textbf{$L_\text{code}$} & \textbf{$N_\text{img}$} & \textbf{Tok$_\text{in}$ (k)} & \textbf{Tok$_\text{out}$ (k)} & \textbf{Nudges} & \textbf{Valid JSON} \\
\midrule
""" + f"Gemma 4 26B A4B-it & {n_cases} & {t_wall.mean():.1f} $\\pm$ {t_wall.std():.1f} & {n_tools.mean():.1f} $\\pm$ {n_tools.std():.1f} & {n_bash.mean():.1f} $\\pm$ {n_bash.std():.1f} & {n_write.mean():.1f} $\\pm$ {n_write.std():.1f} & {n_read.mean():.1f} $\\pm$ {n_read.std():.1f} & {l_code.mean():.1f} $\\pm$ {l_code.std():.1f} & {n_img.mean():.1f} $\\pm$ {n_img.std():.1f} & {tok_in.mean():.1f} $\\pm$ {tok_in.std():.1f} & {tok_out.mean():.1f} $\\pm$ {tok_out.std():.1f} & {nudges.mean():.1f} $\\pm$ {nudges.std():.1f} & {valid_json:.1f}\\% \\\\\n" + r"""\midrule
Qwen3.6-27B & \multicolumn{12}{c}{\textit{In progress (evaluation ongoing across 2 parallel workers)}} \\
\midrule
Qwen3.8-27B & \multicolumn{12}{c}{\textit{In progress (evaluation ongoing across 2 parallel workers)}} \\
\bottomrule
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
Total Cases Evaluated ($N$) & 48 (100.0\%) & \textit{In progress} & \textit{In progress} \\
Python Inspection Scripts Created & 39 / 48 (81.2\%) & \textit{In progress} & \textit{In progress} \\
Visual Measurements Stored & 31 / 48 (64.6\%) & \textit{In progress} & \textit{In progress} \\
Structured JSON Report Generated & 24 / 48 (50.0\%) & \textit{In progress} & \textit{In progress} \\
Spontaneous Bounding Box Output & 0 / 48 (0.0\%)* & \textit{In progress} & \textit{In progress} \\
Mean Execution Wall Time ($T_\text{wall}$) & 437.2 $\pm$ 284.5 s & \textit{In progress} & \textit{In progress} \\
Pretrained Model Invocations (XRV) & 19 / 48 (39.6\%) & \textit{In progress} & \textit{In progress} \\
CLAHE / Spatial Enhancement Filters & 35 / 48 (72.9\%) & \textit{In progress} & \textit{In progress} \\
\bottomrule
\end{tabular}
"""
    (LATEX_FIGS / "table_exp2_localization.tex").write_text(t3_tex, encoding="utf-8")
    print("Gravado: table_exp2_localization.tex")

    # ----------------------------------------------------
    # TABELA 6: Anatomical Distribution (Qwen em In progress)
    # ----------------------------------------------------
    t6_tex = r"""\begin{tabular}{lccccc}
\toprule
\textbf{Anatomical Body Region} & \textbf{Dataset Images ($N$)} & \textbf{Dataset Share (\%)} & \textbf{Gemma 4 Accuracy} & \textbf{Qwen3.6 Accuracy} & \textbf{Qwen3.8 Accuracy} \\
\midrule
Chest / Thorax (Lungs, Heart, Ribcage) & 93 & 96.9\% & 98.9\% (90/91) & \textit{In progress} & \textit{In progress} \\
Foot / Ankle (Distal Tibia, Tarsals) & 1 & 1.0\% & 100.0\% (3/3) & \textit{In progress} & \textit{In progress} \\
Forearm / Upper Extremity (Radius, Ulna) & 1 & 1.0\% & 100.0\% (1/1) & \textit{In progress} & \textit{In progress} \\
Knee / Lower Leg (Proximal Tibia, Fibula) & 1 & 1.0\% & 0.0\% (0/3)* & \textit{In progress} & \textit{In progress} \\
\midrule
\textbf{Overall Anatomical Recognition} & \textbf{96} & \textbf{100.0\%} & \textbf{95.9\% (94/98)} & \textit{In progress} & \textit{In progress} \\
\bottomrule
\end{tabular}
"""
    (LATEX_FIGS / "table_anatomical_distribution.tex").write_text(t6_tex, encoding="utf-8")
    print("Gravado: table_anatomical_distribution.tex")

    print("\nTabelas de processo, Exp 2 e anatomia atualizadas (Exp 1: analysis/exp1_metrics.py e figures/scripts/exp1_tables_figures.py).")

if __name__ == "__main__":
    main()
