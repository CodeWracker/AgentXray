#!/usr/bin/env python3
"""Avalia a acuracia do reconhecimento anatomico / regiao do corpo pelos modelos.

Analisa provenance/first_look.md e os relatorios finais (findings, impression)
para determinar se o modelo identificou corretamente a regiao anatomica sem que
o prompt informasse o que era a imagem.
"""

import collections
import glob
import json
import pathlib
import re
import sys

EVAL = pathlib.Path(__file__).resolve().parent.parent

# Ground truth das imagens avaliadas
def get_ground_truth_region(img_name: str) -> str:
    if img_name == "ID001-xray.png":
        return "Foot / Ankle"
    elif img_name == "ID002-xray.png":
        return "Knee / Lower Leg"
    elif img_name == "ID003-xray.png":
        return "Forearm / Upper Extremity"
    else:
        # Todas as imagens do NIH ChestX-ray14 sao de torax
        return "Chest / Thorax"

CHEST_KEYWORDS = [
    "chest", "thorax", "thoracic", "lung", "pulmonary", "rib", "cxr", 
    "cardiomegaly", "pleural", "pneumothorax", "atelectasis", "consolidation"
]
FOOT_ANKLE_KEYWORDS = [
    "foot", "ankle", "calcaneus", "talus", "tarsal", "metatarsal", "lower extremity", "lower leg"
]
KNEE_LEG_KEYWORDS = [
    "knee", "patella", "lower leg", "tibia", "fibula", "femur", "lower extremity"
]
ARM_FOREARM_KEYWORDS = [
    "forearm", "arm", "radius", "ulna", "wrist", "elbow", "upper extremity"
]

def classify_extracted_text(text: str) -> str:
    lower = text.lower()
    
    # Extrai regiao explicitamente se houver secao Body Region
    m = re.search(r"body\s*region[\s\:\*\#\-]+([^\n\r]+)", lower)
    region_str = m.group(1).strip() if m else lower[:500]
    
    if any(k in region_str for k in ["chest", "thorax", "thoracic", "cxr", "lung"]):
        return "Chest / Thorax"
    if any(k in region_str for k in ["forearm", "wrist", "elbow", "upper extremity", "radius", "ulna"]):
        return "Forearm / Upper Extremity"
    if any(k in region_str for k in ["ankle", "foot", "calcaneus", "talus", "tarsal"]):
        return "Foot / Ankle"
    if any(k in region_str for k in ["knee", "patella"]):
        return "Knee / Lower Leg"
    if any(k in region_str for k in ["lower extremity", "lower leg", "tibia", "fibula"]):
        return "Foot / Ankle" if ("ankle" in lower or "foot" in lower) else "Knee / Lower Leg"
        
    return "Unknown"

def main():
    results_dir = EVAL / "results"
    
    models = ["gemma4-26b", "qwen3.6-27b", "qwen3.8-27b"]
    
    dataset_distribution = collections.Counter()
    model_recognition = collections.defaultdict(lambda: {
        "total": 0,
        "correct": 0,
        "by_region": collections.defaultdict(lambda: {"total": 0, "correct": 0, "predicted": collections.Counter()})
    })

    # Imagens unicas avaliadas
    all_runs = list(results_dir.glob("**/harness_result.json"))
    seen_images = set()

    print(f"Analisando {len(all_runs)} rodadas concluidas...")

    for hr in all_runs:
        run_dir = hr.parent
        manifest_file = run_dir / "run_manifest.json"
        if not manifest_file.exists():
            continue
        try:
            mdata = json.loads(manifest_file.read_text(encoding="utf-8"))
        except Exception:
            continue
            
        model = mdata.get("model")
        if model not in models:
            continue
            
        imgs = mdata.get("images", [])
        if not imgs:
            continue
        img_name = imgs[0].get("file", run_dir.name)
        
        gt_region = get_ground_truth_region(img_name)
        dataset_distribution[img_name] = gt_region
        
        # Le o first_look.md ou relatorio final
        fl_file = list(run_dir.glob("**/provenance/first_look.md"))
        json_file = list(run_dir.glob("*_analysis/*.json"))
        
        text_to_eval = ""
        if fl_file and fl_file[0].exists():
            text_to_eval = fl_file[0].read_text(encoding="utf-8", errors="ignore")
        elif json_file and json_file[0].exists():
            try:
                jd = json.loads(json_file[0].read_text(encoding="utf-8"))
                text_to_eval = f"{jd.get('findings', '')} {jd.get('impression', '')}"
            except Exception:
                pass
                
        if not text_to_eval:
            continue
            
        predicted_region = classify_extracted_text(text_to_eval)
        
        # Considera correto se bate a regiao macro
        is_correct = False
        if gt_region == "Chest / Thorax" and predicted_region == "Chest / Thorax":
            is_correct = True
        elif gt_region == "Foot / Ankle" and predicted_region in ["Foot / Ankle", "Knee / Lower Leg"]:
            # Reconheceu membro inferior corretamente
            is_correct = True
        elif gt_region == "Knee / Lower Leg" and predicted_region in ["Knee / Lower Leg", "Foot / Ankle"]:
            is_correct = True
        elif gt_region == "Forearm / Upper Extremity" and predicted_region == "Forearm / Upper Extremity":
            is_correct = True
            
        st = model_recognition[model]
        st["total"] += 1
        if is_correct:
            st["correct"] += 1
        st["by_region"][gt_region]["total"] += 1
        if is_correct:
            st["by_region"][gt_region]["correct"] += 1
        st["by_region"][gt_region]["predicted"][predicted_region] += 1

    # Conta distribuicao de imagens unicas no dataset avaliado
    region_img_count = collections.Counter(dataset_distribution.values())

    print("\n" + "="*50)
    print("DISTRIBUICAO DE REGIOES ANATOMICAS NO DATASET AVALIADO")
    print("="*50)
    total_imgs = len(dataset_distribution)
    for reg, count in region_img_count.most_common():
        pct = (count / total_imgs) * 100
        print(f"  {reg:28s}: {count:4d} imagens ({pct:5.1f}%)")
    print(f"  {'TOTAL':28s}: {total_imgs:4d} imagens (100.0%)")

    print("\n" + "="*50)
    print("ACURACIA DE RECONHECIMENTO DE REGIAO DO CORPO POR MODELO")
    print("="*50)
    for model in models:
        st = model_recognition[model]
        tot = st["total"]
        if tot == 0:
            print(f"{model}: Nenhuma rodada analisada.")
            continue
        corr = st["correct"]
        acc = (corr / tot) * 100
        print(f"\n--- {model} (Total de rodadas analisadas: {tot}) ---")
        print(f"  Acuracia Geral: {acc:.1f}% ({corr}/{tot})")
        for reg in sorted(st["by_region"].keys()):
            rst = st["by_region"][reg]
            r_tot = rst["total"]
            r_corr = rst["correct"]
            r_acc = (r_corr / r_tot) * 100 if r_tot else 0.0
            print(f"    {reg:26s}: {r_acc:5.1f}% ({r_corr}/{r_tot}) | Predicoes: {dict(rst['predicted'])}")

if __name__ == "__main__":
    main()
