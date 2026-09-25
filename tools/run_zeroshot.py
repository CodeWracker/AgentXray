#!/usr/bin/env python3
"""Executa a baseline zero-shot (sem ferramentas, sem agente, sem loop de execucao) para comparacao.

Envia a imagem diretamente ao modelo multimodal com um único prompt estruturado
e avalia o diagnostico diferencial emitido usando a mesma ontologia clinica (Hit@1, Hit@3, MRR).

Uso:
    tools/py tools/run_zeroshot.py --model gemma4-26b --manifest inputs/benchmarks/exp1_image_level/manifest.json
"""

import argparse
import base64
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request

EVAL = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EVAL / "analysis"))
from align_diagnosis import evaluate_differential_diagnosis

# quando o DGX_UFSC_BASE_URL e um tunel (ngrok) que pode estourar cota de banda, cai para o
# LiteLLM local, alcancavel diretamente no mesmo host da DGX
LOCAL_LITELLM_FALLBACK = "http://127.0.0.1:4000"
THINKING_BUDGET = 6000
ANSWER_BUDGET = 4000


def encode_image(img_path: pathlib.Path) -> str:
    return base64.b64encode(img_path.read_bytes()).decode("utf-8")


# tokens da ultima resposta (campo usage do LiteLLM), gravados por caso em _usage/
LAST_USAGE: dict = {}


def closed_vocabulary(version: str) -> bool:
    # a condicao de vocabulario fechado usa as pastas de versao terminadas em "c" (prompts/v4c, results/v4c)
    return version.endswith("c")


def vocabulary_text() -> str:
    from make_closed_prompts import EXP1_LABELS
    return ("Every entry of \"differential_diagnosis\" must be exactly one of the following labels, written as shown, "
            "with no other words; use No Finding only when no label applies, and do not repeat a label: "
            + ", ".join(EXP1_LABELS) + ".\n")


def classifier_text(pred: dict, weights: str) -> str:
    ranked = sorted(pred["scores"].items(), key=lambda kv: kv[1], reverse=True)
    return ("An automated classifier was run on this image before you: a DenseNet-121 from the TorchXRayVision library "
            f"(weights {weights}). It outputs one score per label between 0 and 1, where higher means more likely and "
            "0.5 is the classifier's operating threshold. It can be wrong. Its scores, from highest to lowest: "
            + "; ".join(f"{c.replace('_', ' ')} {v:.2f}" for c, v in ranked)
            + ". Use them together with your own inspection of the image.\n")


def query_zeroshot(image_path: pathlib.Path, model: str, base_url: str, api_key: str, closed: bool = False,
                   hint: str = "") -> dict:
    b64_img = encode_image(image_path)
    primary = ("Primary hypothesis (one label from the list below)" if closed
               else "Primary hypothesis (most likely condition or Normal / No acute finding)")
    prompt_text = (
        f"You are analyzing this medical image `{image_path.name}` as a second-opinion support tool for a medical professional. "
        "You do not have access to any external tools or code execution. Inspect the image directly with your vision "
        "and output ONLY a valid JSON object with exactly these fields:\n"
        "{\n"
        f'  "image": "{image_path.name}",\n'
        '  "findings": "Detailed description of visible radiological findings.",\n'
        '  "impression": "Overall primary clinical impression.",\n'
        '  "differential_diagnosis": [\n'
        f'    "{primary}",\n'
        '    "Secondary hypothesis",\n'
        '    "Tertiary hypothesis"\n'
        "  ],\n"
        '  "limitations": "Technical and projection limitations."\n'
        "}\n"
        + hint
        + (vocabulary_text() if closed else "")
        + "Do not include any conversational filler. Output only the JSON block."
    )

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}},
                ],
            }
        ],
        "temperature": 0.0,
        # orcamento de pensamento: o servidor encerra o raciocinio em THINKING_BUDGET tokens e o modelo ainda tem
        # ANSWER_BUDGET tokens para a resposta; um max_tokens unico cortaria o raciocinio antes do JSON
        "thinking_token_budget": THINKING_BUDGET,
        "max_tokens": THINKING_BUDGET + ANSWER_BUDGET,
    }

    urls_to_try = [base_url] if base_url == LOCAL_LITELLM_FALLBACK else [base_url, LOCAL_LITELLM_FALLBACK]
    res, last_error = None, None
    for url in urls_to_try:
        req = urllib.request.Request(
            f"{url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=1800) as resp:
                res = json.loads(resp.read().decode("utf-8"))
            break
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            last_error = e
    if res is None:
        raise last_error

    LAST_USAGE.clear()
    LAST_USAGE.update(res.get("usage") or {})
    LAST_USAGE["finish_reason"] = res["choices"][0].get("finish_reason")
    LAST_USAGE["thinking_token_budget"] = THINKING_BUDGET
    msg = res["choices"][0]["message"]
    # alguns modelos devolvem a resposta so no campo de raciocinio
    content = msg.get("content") or msg.get("reasoning_content") or ""
    cleaned = content.strip()
    if "```json" in cleaned:
        parts = cleaned.split("```json")
        if len(parts) > 1:
            cleaned = parts[1].split("```")[0]
    elif "```" in cleaned:
        parts = cleaned.split("```")
        if len(parts) > 1:
            cleaned = parts[1]
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except Exception:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(cleaned[start : end + 1])
            except Exception:
                pass
        raise ValueError(f"Resposta nao contem JSON valido: {content[:300]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--base-url", default=os.environ.get("DGX_UFSC_BASE_URL", "http://localhost:4000"))
    parser.add_argument("--version", default="v4", help="pasta da condicao em results/; terminada em c = vocabulario fechado")
    parser.add_argument("--classifier", help="pesos do TorchXRayVision cujas saidas entram no prompt (degrau modelo + classificador, "
                        "sem ferramentas); le results/supervised/exp1_image_level/<pesos>/predictions.json")
    args = parser.parse_args()
    closed = closed_vocabulary(args.version)
    preds = None
    if args.classifier:
        pred_file = EVAL / "results" / "supervised" / "exp1_image_level" / args.classifier / "predictions.json"
        preds = json.loads(pred_file.read_text(encoding="utf-8"))["predictions"]

    args.base_url = args.base_url.rstrip("/")
    api_key = os.environ.get("DGX_UFSC_API_KEY", "")
    manifest_file = args.manifest.resolve()
    with open(manifest_file, encoding="utf-8") as f:
        items = json.load(f)

    if args.max_cases:
        items = items[: args.max_cases]

    bench_name = manifest_file.parent.name
    mode_dir = f"zeroshot_clf-{args.classifier}" if args.classifier else "zeroshot"
    out_dir = EVAL / "results" / args.version / mode_dir / bench_name / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    nih_images = EVAL / "inputs" / "nih_chestxray" / "images"

    results = []
    print(f"Iniciando avaliacao Zero-Shot para {args.model} no benchmark {bench_name} ({len(items)} casos)...")

    for idx, it in enumerate(items, 1):
        img_name = it["image"]
        img_path = nih_images / img_name
        if not img_path.exists():
            cand = list(EVAL.glob(f"**/{img_name}"))
            if cand:
                img_path = cand[0]
            else:
                print(f"[{idx}/{len(items)}] Imagem {img_name} nao encontrada. Pulando.")
                continue

        out_json_path = out_dir / f"{img_path.stem}.json"
        if out_json_path.exists():
            try:
                parsed = json.loads(out_json_path.read_text(encoding="utf-8"))
                dx = parsed.get("differential_diagnosis", [])
                eval_res = evaluate_differential_diagnosis(dx, it.get("finding_labels", []))
                results.append(eval_res)
                print(f"[{idx}/{len(items)}] {img_name}: (ja existente) Hit@1={eval_res['hit_at_1']}, Hit@3={eval_res['hit_at_3']}, MRR={eval_res['mrr']}")
                continue
            except Exception:
                pass

        t0 = time.time()
        try:
            hint = classifier_text(preds[img_name], args.classifier) if preds is not None else ""
            parsed = query_zeroshot(img_path, args.model, args.base_url, api_key, closed=closed, hint=hint)
            out_json_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False) + "\n")
            usage_dir = out_dir / "_usage"
            usage_dir.mkdir(exist_ok=True)
            (usage_dir / out_json_path.name).write_text(json.dumps({"elapsed_s": round(time.time() - t0, 2), "usage": dict(LAST_USAGE)}, indent=2) + "\n")
            dx = parsed.get("differential_diagnosis", [])
            eval_res = evaluate_differential_diagnosis(dx, it.get("finding_labels", []))
            eval_res["elapsed_s"] = round(time.time() - t0, 2)
            results.append(eval_res)
            print(f"[{idx}/{len(items)}] {img_name} ({eval_res['elapsed_s']}s): Hit@1={eval_res['hit_at_1']}, Hit@3={eval_res['hit_at_3']}, MRR={eval_res['mrr']}")
        except Exception as e:
            print(f"[{idx}/{len(items)}] {img_name}: Erro: {e}")

    if not args.max_cases:
        # marca a passada completa: sem isso a analise nao distingue caso que falhou de caso nao executado
        (out_dir / "_complete.json").write_text(json.dumps({"cases": len(items), "written": len(list(out_dir.glob("0*.json")))}) + "\n")

    if results:
        n = len(results)
        h1 = sum(1 for r in results if r["hit_at_1"])
        h3 = sum(1 for r in results if r["hit_at_3"])
        mrr = sum(r["mrr"] for r in results) / n
        print(f"\n{'='*40}")
        print(f"RESUMO ZERO-SHOT: {args.model} ({n} casos)")
        print(f"  Hit@1: {h1 / n * 100:.1f}% ({h1}/{n})")
        print(f"  Hit@3: {h3 / n * 100:.1f}% ({h3}/{n})")
        print(f"  MRR Medio: {mrr:.3f}")
        print(f"{'='*40}\n")


if __name__ == "__main__":
    main()
