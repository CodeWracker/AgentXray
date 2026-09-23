#!/usr/bin/env python3
"""Executa a baseline zero-shot para o Experimento 2 (Localizacao Espacial / Bounding Box).

Envia a imagem diretamente ao modelo multimodal com um prompt estruturado de localizacao,
solicitando a identificacao de lesoes e coordenadas numericas de caixas delimitadoras [x, y, w, h].
Avalia a sobreposicao espacial contra as anotacoes dos radiologistas do NIH (IoU >= 0.1, 0.3, 0.5).

Uso:
    tools/py tools/run_zeroshot_exp2.py --model gemma4-26b --manifest inputs/benchmarks/exp2_bbox_localization/manifest.json
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
from bbox_eval import evaluate_bboxes, extract_predicted_bboxes

# quando o DGX_UFSC_BASE_URL e um tunel (ngrok) que pode estourar cota de banda, cai para o
# LiteLLM local, alcancavel diretamente no mesmo host da DGX
LOCAL_LITELLM_FALLBACK = "http://127.0.0.1:4000"


def encode_image(img_path: pathlib.Path) -> str:
    return base64.b64encode(img_path.read_bytes()).decode("utf-8")


def query_zeroshot_localization(image_path: pathlib.Path, model: str, base_url: str, api_key: str) -> dict:
    b64_img = encode_image(image_path)
    prompt_text = (
        f"You are analyzing this chest radiograph `{image_path.name}` as a medical imaging specialist. "
        "Your objective is to identify and spatially localize any focal thoracic abnormalities using bounding boxes. "
        "Output ONLY a valid JSON object with exactly these fields:\n"
        "{\n"
        f'  "image": "{image_path.name}",\n'
        '  "findings": "Detailed description of visible radiological findings.",\n'
        '  "impression": "Overall primary clinical impression.",\n'
        '  "differential_diagnosis": [\n'
        '    "Primary hypothesis (most likely condition or Normal / No acute finding)",\n'
        '    "Secondary hypothesis"\n'
        '  ],\n'
        '  "localized_lesions": [\n'
        '    {\n'
        '      "label": "Pathology name (e.g., Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax)",\n'
        '      "bbox": [x, y, w, h],\n'
        '      "confidence": "high",\n'
        '      "reasoning": "Visual evidence justifying this bounding box"\n'
        '    }\n'
        '  ],\n'
        '  "limitations": "Technical and projection limitations."\n'
        "}\n"
        "Bounding box rules:\n"
        "- Coordinates [x, y, w, h] must be in original image pixels (0 to 1024), where (x, y) is top-left and (w, h) are width and height.\n"
        "- If no focal abnormality is present, set \"localized_lesions\": [].\n"
        "Do not include conversational filler. Output only the JSON block."
    )

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a specialized thoracic radiology AI. Output only the requested JSON object. Do not include conversational filler or separate markdown text outside JSON.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}},
                ],
            },
        ],
        "temperature": 0.0,
        "max_tokens": 6000,
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
            with urllib.request.urlopen(req, timeout=300) as resp:
                res = json.loads(resp.read().decode("utf-8"))
            break
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            last_error = e
    if res is None:
        raise last_error

    msg = res["choices"][0]["message"]
    content = msg.get("content") or msg.get("reasoning_content") or ""
    cleaned = content.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except Exception:
        start = cleaned.find("{")
        if start != -1:
            try:
                obj, _ = json.JSONDecoder().raw_decode(cleaned[start:])
                return obj
            except Exception:
                pass
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
    args = parser.parse_args()

    args.base_url = args.base_url.rstrip("/")
    api_key = os.environ.get("DGX_UFSC_API_KEY", "")
    manifest_file = args.manifest.resolve()
    with open(manifest_file, encoding="utf-8") as f:
        items = json.load(f)

    if args.max_cases:
        items = items[: args.max_cases]

    bench_name = manifest_file.parent.name
    out_dir = EVAL / "results" / "v2" / "zeroshot" / bench_name / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    nih_images = EVAL / "inputs" / "nih_chestxray" / "images"

    results = []
    print(f"Iniciando avaliacao Zero-Shot Localizacao para {args.model} no benchmark {bench_name} ({len(items)} casos)...")

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
        gt_boxes = it.get("bboxes", [])

        if out_json_path.exists():
            try:
                parsed = json.loads(out_json_path.read_text(encoding="utf-8"))
                pred_boxes = extract_predicted_bboxes(parsed)
                eval_res = evaluate_bboxes(pred_boxes, gt_boxes)
                results.append(eval_res)
                print(f"[{idx}/{len(items)}] {img_name}: (ja existente) pred={len(pred_boxes)}, max_IoU={eval_res['max_iou']}, IoU>=0.1={eval_res['hit_iou_01']}, IoU>=0.3={eval_res['hit_iou_03']}, IoU>=0.5={eval_res['hit_iou_05']}")
                continue
            except Exception:
                pass

        t0 = time.time()
        try:
            parsed = query_zeroshot_localization(img_path, args.model, args.base_url, api_key)
            out_json_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False) + "\n")
            pred_boxes = extract_predicted_bboxes(parsed)
            eval_res = evaluate_bboxes(pred_boxes, gt_boxes)
            eval_res["elapsed_s"] = round(time.time() - t0, 2)
            eval_res["num_pred_boxes"] = len(pred_boxes)
            results.append(eval_res)
            print(f"[{idx}/{len(items)}] {img_name} ({eval_res['elapsed_s']}s): pred={len(pred_boxes)}, max_IoU={eval_res['max_iou']}, IoU>=0.1={eval_res['hit_iou_01']}, IoU>=0.3={eval_res['hit_iou_03']}, IoU>=0.5={eval_res['hit_iou_05']}")
        except Exception as e:
            print(f"[{idx}/{len(items)}] {img_name}: Erro: {e}")

    if not args.max_cases:
        # marca a passada completa: sem isso a analise nao distingue caso que falhou de caso nao executado
        (out_dir / "_complete.json").write_text(json.dumps({"cases": len(items), "written": len(list(out_dir.glob("0*.json")))}) + "\n")

    if results:
        n = len(results)
        hit_01 = sum(1 for r in results if r.get("hit_iou_01"))
        hit_03 = sum(1 for r in results if r.get("hit_iou_03"))
        hit_05 = sum(1 for r in results if r.get("hit_iou_05"))
        mean_iou = sum(r.get("max_iou", 0.0) for r in results) / n
        print(f"\n{'='*40}")
        print(f"RESUMO ZERO-SHOT LOCALIZACAO: {args.model} ({n} casos)")
        print(f"  IoU >= 0.1: {hit_01 / n * 100:.1f}% ({hit_01}/{n})")
        print(f"  IoU >= 0.3: {hit_03 / n * 100:.1f}% ({hit_03}/{n})")
        print(f"  IoU >= 0.5: {hit_05 / n * 100:.1f}% ({hit_05}/{n})")
        print(f"  Max IoU Medio: {mean_iou:.3f}")
        print(f"{'='*40}\n")


if __name__ == "__main__":
    main()
