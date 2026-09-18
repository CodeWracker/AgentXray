#!/usr/bin/env python3
"""Extracao de conteudo clinico (regiao anatomica e presenca de fratura) com regras e negacao.

Le analysis/summary.csv e analysis/reference.json, analisa os textos de findings
e impression, e grava analysis/content_audit.csv com colunas para auditoria manual.

Uso:
    tools/py analysis/content.py [--summary analysis/summary.csv] [--reference analysis/reference.json]
"""

import argparse
import csv
import json
import pathlib
import re
import sys

EVAL = pathlib.Path(__file__).resolve().parent.parent

NEGATION_PATTERNS = [
    r"\bno\s+(?:definite|acute|visible|gross|focal|evident|obvious|displaced|clear)?\s*fracture\b",
    r"\bwithout\s+(?:evidence\s+of|definite|acute|visible)?\s*fracture\b",
    r"\bnegative\s+for\s+(?:acute\s+)?fracture\b",
    r"\brules?\s+out\s+fracture\b",
    r"\bfree\s+of\s+fracture\b",
    r"\babsence\s+of\s+(?:an?\s+)?(?:acute\s+)?fracture\b",
    r"\bdenies\s+fracture\b",
    r"\bno\s+bony\s+(?:injury|fracture)\b",
    r"\bno\s+evidence\s+of\s+(?:an?\s+)?fracture\b",
    r"\bintact\s+cort(?:ex|ices)\b",
    r"\bcortical\s+margins?\s+(?:appear\s+)?continuous\b",
    r"\bsem\s+fratura\b",
    r"\bnao\s+ha\s+fratura\b",
]

POSITIVE_PATTERNS = [
    r"\bfracture\s+of\b",
    r"\btransverse\s+fracture\b",
    r"\boblique\s+fracture\b",
    r"\bcomminuted\s+fracture\b",
    r"\bdisplaced\s+fracture\b",
    r"\bnondisplaced\s+fracture\b",
    r"\bminimally\s+displaced\s+fracture\b",
    r"\bfracture\s+line\b",
    r"\bcortical\s+(?:step-off|disruption|interruption|discontinuity|break)\b",
    r"\bbone\s+fracture\b",
    r"\bconsistent\s+with\s+(?:a\s+)?(?:acute\s+)?fracture\b",
    r"\bfratura\b",
]

REGION_PATTERNS = {
    "upper_extremity": [
        r"\bforearm\b",
        r"\bradius\b",
        r"\bulna\b",
        r"\belbow\b",
        r"\bhumerus\b",
        r"\bwrist\b",
        r"\bcarpal\b",
        r"\bhand\b",
        r"\bantebra[cç]o\b",
        r"\br[aá]dio\b",
        r"\bcotovelo\b",
        r"\bpunho\b",
    ],
    "lower_extremity": [
        r"\bankle\b",
        r"\bfoot\b",
        r"\bcalcaneus\b",
        r"\btalus\b",
        r"\btibia\b",
        r"\bfibula\b",
        r"\bknee\b",
        r"\bfemur\b",
        r"\bpatella\b",
        r"\btornozelo\b",
        r"\bp[eé]\b",
        r"\bcalc[aâ]neo\b",
        r"\bt[ií]bia\b",
        r"\bf[ií]bula\b",
        r"\bjoelho\b",
    ],
    "chest": [
        r"\bchest\b",
        r"\bthorax\b",
        r"\blung\b",
        r"\bpleural\b",
        r"\bcardiac\b",
        r"\bheart\b",
        r"\bmediastinum\b",
        r"\bt[oó]rax\b",
        r"\bpulm[aã]o\b",
    ],
}


def classify_region(text: str) -> str:
    text_lower = text.lower()
    scores = {}
    for region, patterns in REGION_PATTERNS.items():
        count = sum(len(re.findall(p, text_lower)) for p in patterns)
        scores[region] = count

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if sorted_scores[0][1] > 0:
        return sorted_scores[0][0]
    return "indeterminate"


def classify_fracture(text: str) -> tuple[bool | None, list[str], list[str]]:
    text_lower = text.lower()
    neg_matches = []
    for pat in NEGATION_PATTERNS:
        found = re.findall(pat, text_lower)
        if found:
            neg_matches.extend(found)

    pos_matches = []
    for pat in POSITIVE_PATTERNS:
        found = re.findall(pat, text_lower)
        if found:
            pos_matches.extend(found)

    if pos_matches and not neg_matches:
        return True, pos_matches, neg_matches
    if neg_matches and not pos_matches:
        return False, pos_matches, neg_matches
    if pos_matches and neg_matches:
        # Se ha afirmacao explicita de fratura no impression, verificar contexto
        # Ex: "Findings are most consistent with fracture... No other fracture seen"
        if any(term in text_lower for term in ["consistent with a fracture", "transverse fracture", "cortical step-off"]):
            return True, pos_matches, neg_matches
        return False, pos_matches, neg_matches

    if "fracture" in text_lower:
        return False, pos_matches, neg_matches
    return None, pos_matches, neg_matches


def main():
    parser = argparse.ArgumentParser(description="Extrai conteudo clinico e prepara planilha de auditoria.")
    parser.add_argument("--summary", type=pathlib.Path, default=EVAL / "analysis" / "summary.csv")
    parser.add_argument("--reference", type=pathlib.Path, default=EVAL / "analysis" / "reference.json")
    parser.add_argument("--out", type=pathlib.Path, default=EVAL / "analysis" / "content_audit.csv")
    args = parser.parse_args()

    if not args.summary.exists():
        print("Arquivo summary.csv nao encontrado em", args.summary, file=sys.stderr)
        return 1

    reference = {}
    if args.reference.exists():
        reference = json.loads(args.reference.read_text(encoding="utf-8"))

    with open(args.summary, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    audit_rows = []
    for row in rows:
        findings = row.get("findings", "")
        impression = row.get("impression", "")
        combined_text = f"{impression}\n{findings}"

        pred_region = classify_region(combined_text)
        pred_fracture, pos_cues, neg_cues = classify_fracture(impression if impression.strip() else findings)

        image = row.get("image", "")
        ref_data = reference.get(image, {})
        ref_region = ref_data.get("body_region")
        ref_fracture = ref_data.get("fracture_reference")

        region_match = (pred_region == ref_region) if ref_region else None
        fracture_match = (pred_fracture == ref_fracture) if ref_fracture is not None else None

        audit_rows.append({
            "condition": row.get("condition"),
            "prompt_version": row.get("prompt_version"),
            "mode": row.get("mode"),
            "model": row.get("model"),
            "run_id": row.get("run_id"),
            "image": image,
            "run_dir": row.get("run_dir"),
            "predicted_region": pred_region,
            "reference_region": ref_region,
            "region_match": region_match,
            "predicted_fracture": pred_fracture,
            "reference_fracture": ref_fracture,
            "fracture_match": fracture_match,
            "positive_cues": "; ".join(set(pos_cues)),
            "negation_cues": "; ".join(set(neg_cues)),
            "impression_snippet": impression[:300].replace("\n", " "),
            "manual_audit_confirmed": "",
            "manual_audit_notes": "",
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if audit_rows:
        fieldnames = list(audit_rows[0].keys())
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(audit_rows)
        print(f"Gerado {args.out} com {len(audit_rows)} linhas para auditoria")
    return 0


if __name__ == "__main__":
    sys.exit(main())
