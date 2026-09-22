#!/usr/bin/env python3
"""Testes da projecao de hipoteses e da pontuacao do Exp 1. Uso: tools/py analysis/test_exp1_scoring.py"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from align_diagnosis import score_case  # noqa: E402
from clinical_ontology import ONTOLOGY, project_hypothesis  # noqa: E402

PROJECTION = [
    ("No pneumothorax or pleural effusion", [], ["Pneumothorax", "Effusion"]),
    ("No pneumothorax/effusion", [], ["Pneumothorax", "Effusion"]),
    ("Pneumonia with parapneumonic effusion", ["Pneumonia", "Effusion"], []),
    ("Massive left pleural effusion", ["Effusion"], []),
    ("Pneumothorax cannot be excluded", ["Pneumothorax"], []),
    ("Occult pulmonary nodule cannot be entirely excluded on a single frontal view", ["Nodule"], []),
    ("Consolidation / pneumonia (with or without parapneumonic effusion)", ["Consolidation", "Pneumonia", "Effusion"], []),
    ("Normal / No acute cardiopulmonary finding aside from hernia", ["No Finding", "Hernia"], []),
    ("No acute cardiopulmonary disease", ["No Finding"], []),
    ("Absence of vascular markings at the apex", ["Pneumothorax"], []),
    ("Nipple shadow (less likely given absence of pleural effusion)", [], ["Effusion"]),
    ("Pulmonary edema, pneumonia ruled out", ["Edema"], ["Pneumonia"]),
    ("Retrocardiac mass with air-fluid level", ["Hernia"], []),
    ("Bilateral infiltrates", ["Infiltration"], []),
    ("Cardiomegaly without pulmonary edema", ["Cardiomegaly"], ["Edema"]),
    ("Rib fracture", [], []),
]


def main() -> int:
    failures = 0
    for text, classes, negated in PROJECTION:
        got = project_hypothesis(text)
        if got["classes"] != classes or got["negated"] != negated:
            failures += 1
            print(f"FALHA projecao: {text!r} -> {got}")
    assert not project_hypothesis("N/A")["informative"]

    # o baseline escreve cada classe pelo primeiro sinonimo, que precisa voltar para a mesma classe
    for cls, info in ONTOLOGY.items():
        if project_hypothesis(info["synonyms"][0])["classes"] != [cls]:
            failures += 1
            print(f"FALHA sinonimo canonico: {cls}")

    # caso misto da discussao: 2 OOV e 1 no vocabulario mas errado
    s = score_case(["Rib fracture", "ARDS", "Cardiomegaly"], ["Effusion", "Atelectasis"])
    assert s["category"] == "partial_oov_miss" and abs(s["oov_rate"] - 2 / 3) < 1e-9 and s["precision3"] == 0.0

    # hipotese composta ocupa uma unica posicao de rank
    s = score_case(["Pneumonia with parapneumonic effusion", "Atelectasis", "Mass"], ["Atelectasis"])
    assert s["rank"] == 2 and s["hit1"] is False and s["hit3"] is True and s["mrr3"] == 0.5
    assert s["recall3"] == 1.0 and abs(s["precision3"] - 1 / 4) < 1e-9

    # janela K=3, itens sem conteudo nao ocupam posicao, Hit@1 filtrado ignora OOV na frente
    s = score_case(["N/A", "ARDS", "Effusion", "Mass", "Atelectasis"], ["Effusion", "Atelectasis"])
    assert [h["text"] for h in s["window"]] == ["ARDS", "Effusion", "Mass"]
    assert s["hit1"] is False and s["hit1_f"] is True and s["recall3"] == 0.5

    # diferencial vazio e totalmente OOV
    assert score_case([], ["Mass"])["category"] == "empty"
    s = score_case(["ARDS", "Rib fracture"], ["Mass"])
    assert s["category"] == "total_oov" and s["precision3"] is None and s["f1"] == 0.0

    print("ok" if failures == 0 else f"{failures} falhas")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
