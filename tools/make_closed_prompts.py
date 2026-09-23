#!/usr/bin/env python3
"""Gera a condicao de vocabulario fechado a partir dos prompts abertos: prompts/<versao>/ -> prompts/<versao>c/.

A condicao aberta nunca revela classes nem regiao. A fechada existe so para comparar nossos metodos com
classificadores e baselines de conjunto fechado nas mesmas condicoes: ela copia os prompts abertos sem mudar mais
nada e acrescenta a lista de rotulos permitidos, antes da validacao final. Rodar de novo sobrescreve a pasta gerada.

Uso:
    tools/py tools/make_closed_prompts.py [--src v4]
"""

import argparse
import pathlib
import shutil

PROMPTS = pathlib.Path(__file__).resolve().parent.parent / "prompts"

# rotulos do ChestX-ray14 (Pleural_Thickening escrito como texto) e os 8 do subconjunto com caixas
EXP1_LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion", "Emphysema", "Fibrosis", "Hernia",
               "Infiltration", "Mass", "Nodule", "Pleural Thickening", "Pneumonia", "Pneumothorax", "No Finding"]
EXP2_LABELS = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax"]

OPEN_PRIMARY = "\"Primary hypothesis (most likely condition or 'Normal / No acute finding')\""
CLOSED_PRIMARY = "\"Primary hypothesis (one label from the ANSWER VOCABULARY section)\""
OPEN_BOX_LABEL = "\"label\": \"Name of the finding, in your own words\""
CLOSED_BOX_LABEL = "\"label\": \"One label from the ANSWER VOCABULARY section\""


def vocabulary_section(exp2: bool) -> str:
    text = ("# ANSWER VOCABULARY\n\n"
            "Every entry of `\"differential_diagnosis\"` must be exactly one of the following labels, written as shown, "
            "with no other words. Use `No Finding` only when no label applies. Do not repeat a label.\n\n"
            + "\n".join(f"* `{label}`" for label in EXP1_LABELS) + "\n")
    if exp2:
        text += ("\nEvery `\"label\"` in `\"localized_lesions\"` must be exactly one of the following labels, written as "
                 "shown:\n\n" + "\n".join(f"* `{label}`" for label in EXP2_LABELS) + "\n")
    return text + "\n"


def close_prompt(text: str, exp2: bool) -> str:
    assert OPEN_PRIMARY in text, "exemplo do schema mudou; atualize make_closed_prompts.py"
    text = text.replace(OPEN_PRIMARY, CLOSED_PRIMARY)
    if exp2:
        assert OPEN_BOX_LABEL in text
        text = text.replace(OPEN_BOX_LABEL, CLOSED_BOX_LABEL)
    marker = next(line for line in text.splitlines() if line.startswith("# ") and "FINAL VALIDATION" in line)
    return text.replace(marker, vocabulary_section(exp2) + marker, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="v4")
    args = ap.parse_args()
    src, dst = PROMPTS / args.src, PROMPTS / f"{args.src}c"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    for name in ("single.md", "agents.md", "single_exp2.md", "agents_exp2.md"):
        path = dst / name
        path.write_text(close_prompt(path.read_text(encoding="utf-8"), exp2="exp2" in name), encoding="utf-8")
    (dst / "README.md").write_text(
        f"Gerado por tools/make_closed_prompts.py a partir de prompts/{args.src}/ (condicao de vocabulario fechado). "
        "Nao edite a mao: edite os prompts abertos e gere de novo.\n", encoding="utf-8")
    print(f"{src} -> {dst}")


if __name__ == "__main__":
    main()
