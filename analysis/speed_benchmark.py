#!/usr/bin/env python3
"""Velocidade de cada modelo medida nas sessoes reais das rodadas, sem mandar nenhuma requisicao aos servidores.

Um benchmark ativo disputaria o modelo com as filas (proibido pelas regras do projeto) e distorceria as duas
medicoes. Cada mensagem do assistente numa sessao exportada do opencode e uma chamada ao modelo, com tokens de
entrada e saida e os instantes de inicio e fim. Com o cache de prefixo do vLLM, so a parte nova do contexto passa
pelo prefill; ela e estimada como a entrada da chamada menos o contexto da chamada anterior da mesma sessao.

Por modelo, ajusta por minimos quadrados nao negativos

    duracao = latencia fixa + prefill_novo / taxa_prefill + saida / taxa_decode

e reporta tambem a taxa bruta saida/duracao por faixa de tamanho do contexto (o decode fica mais lento com o
contexto maior). O streaming nao registra o primeiro token, entao a separacao entre prefill e decode e estimada.

Uso:
    tools/py analysis/speed_benchmark.py
Saida:
    analysis/speed/speed.json e speed_calls.csv
"""

import csv
import json
import pathlib

import numpy as np
from scipy.optimize import nnls

EVAL = pathlib.Path(__file__).resolve().parent.parent
RESULTS = EVAL / "results"
OUT = EVAL / "analysis" / "speed"
MODELS = ["gemma4-26b", "qwen3.6-27b", "qwen3.8-27b"]
BINS = [(0, 16_000), (16_000, 32_000), (32_000, 64_000), (64_000, 128_000), (128_000, 10**9)]


def calls(model: str) -> list[dict]:
    rows = []
    for f in sorted(RESULTS.glob(f"v4*/*/*/{model}/*/sessions/*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        prev_ctx = 0
        for m in data.get("messages", []):
            info = m.get("info", {})
            tok, t = info.get("tokens") or {}, info.get("time") or {}
            if info.get("role") != "assistant" or not t.get("completed") or not tok:
                continue
            inp = tok.get("input", 0) + (tok.get("cache") or {}).get("read", 0)
            out = tok.get("output", 0) + tok.get("reasoning", 0)
            dur = (t["completed"] - t["created"]) / 1000
            if dur <= 0 or out <= 0:
                continue
            rows.append({"model": model, "session": f.stem, "run": f.parent.parent.name, "input": inp,
                         "new_input": max(inp - prev_ctx, 0), "output": out, "duration_s": dur})
            prev_ctx = inp + out
    return rows


def fit(rows: list[dict]) -> dict:
    a = np.array([[1.0, r["new_input"], r["output"]] for r in rows])
    y = np.array([r["duration_s"] for r in rows])
    coef, _ = nnls(a, y)
    pred = a @ coef
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return {"fixed_latency_s": coef[0], "prefill_tok_s": 1 / coef[1] if coef[1] > 0 else None,
            "decode_tok_s": 1 / coef[2] if coef[2] > 0 else None, "r2": r2}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary, all_rows = {}, []
    for model in MODELS:
        rows = calls(model)
        all_rows += rows
        if not rows:
            continue
        # taxa bruta so em chamadas com saida suficiente para diluir o prefill e a latencia fixa
        long = [r for r in rows if r["output"] >= 200]
        by_bin = {}
        for lo, hi in BINS:
            sel = [r["output"] / r["duration_s"] for r in long if lo <= r["input"] < hi]
            if sel:
                by_bin[f"{lo // 1000}k-{hi // 1000 if hi < 10**9 else 'max'}k"] = {
                    "calls": len(sel), "median_tok_s": float(np.median(sel))}
        summary[model] = {
            "calls": len(rows), "runs": len({r["run"] for r in rows}),
            "median_call_s": float(np.median([r["duration_s"] for r in rows])),
            "median_output_tokens": float(np.median([r["output"] for r in rows])),
            "median_context_tokens": float(np.median([r["input"] for r in rows])),
            "raw_output_tok_s_by_context": by_bin,
            "fit": fit(rows),
        }
    (OUT / "speed.json").write_text(json.dumps(summary, indent=2) + "\n")
    with open(OUT / "speed_calls.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)
    for model, s in summary.items():
        ft = s["fit"]
        print(f"{model}: {s['calls']} chamadas em {s['runs']} rodadas | chamada mediana {s['median_call_s']:.1f} s, "
              f"saida mediana {s['median_output_tokens']:.0f} tok, contexto mediano {s['median_context_tokens']:.0f} tok")
        print(f"   ajuste: decode {ft['decode_tok_s'] or float('nan'):.1f} tok/s, prefill {ft['prefill_tok_s'] or float('nan'):.0f} tok/s, "
              f"latencia fixa {ft['fixed_latency_s']:.2f} s, R2 {ft['r2']:.2f}")
        for b, v in s["raw_output_tok_s_by_context"].items():
            print(f"   contexto {b:<10} {v['calls']:>5} chamadas  {v['median_tok_s']:.1f} tok/s (saida/duracao)")


if __name__ == "__main__":
    main()
