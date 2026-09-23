#!/usr/bin/env python3
"""Teste empirico de vazamento: os pesos do TorchXRayVision que viram o ChestX-ray14 viram a particao oficial de teste?

Nao ha como saber pela documentacao instalada qual particao do NIH entrou no treino. O teste compara, para cada
classe, a AUC numa amostra da particao oficial de treino (train_val_list.txt) com a AUC numa amostra da particao
oficial de teste (test_list.txt):

    gap = AUC(treino oficial) - AUC(teste oficial)

Modelos que nunca viram o NIH (chex, pc, mimic_nb, mimic_ch) medem quanto a particao de teste e mais dificil por si
so. Modelos que viram o NIH (all, nih) somam a isso a memorizacao das imagens de treino. Se o treino respeitou a
particao oficial, o gap dos modelos que viram o NIH e maior que o dos controles (diferenca das diferencas > 0).
Se o treino usou uma divisao aleatoria sobre o dataset inteiro, as duas particoes foram vistas na mesma proporcao e
a diferenca das diferencas fica perto de zero. Os intervalos vem de bootstrap por paciente.

Uso:
    tools/py analysis/txv_leakage_check.py [--n 3000] [--threads 16]
Saida:
    analysis/leakage/txv_leakage.json e txv_leakage.csv
"""

import argparse
import csv
import json
import pathlib

import numpy as np
import skimage.io
import torch
import torch.nn.functional as F
import torchxrayvision as xrv
from scipy.stats import rankdata

EVAL = pathlib.Path(__file__).resolve().parent.parent
NIH = EVAL / "inputs" / "nih_chestxray"
OUT = EVAL / "analysis" / "leakage"
SEEN = ["densenet121-res224-all", "densenet121-res224-nih"]
UNSEEN = ["densenet121-res224-pc", "densenet121-res224-chex", "densenet121-res224-mimic_nb", "densenet121-res224-mimic_ch"]
# classes presentes em todos os modelos acima
COMMON = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion", "Pneumonia", "Pneumothorax"]


def load_image(path: pathlib.Path) -> torch.Tensor:
    # mesmo pre-processamento da baseline supervisionada (analysis/eval_supervised_densenet.py)
    img = xrv.datasets.normalize(skimage.io.imread(str(path)), 255)
    if img.ndim > 2:
        img = img[:, :, 0]
    return F.interpolate(torch.from_numpy(img[None, None, ...]).float(), size=(224, 224), mode="bilinear", align_corners=False)[0]


def sample(names: set, rows: dict, n: int, rng) -> list:
    # amostra por paciente: uma imagem por paciente, para o bootstrap por paciente ficar simples
    by_patient = {}
    for name in sorted(names):
        by_patient.setdefault(rows[name]["Patient ID"], []).append(name)
    patients = sorted(by_patient)
    chosen = rng.choice(len(patients), size=min(n, len(patients)), replace=False)
    return [by_patient[patients[i]][rng.integers(len(by_patient[patients[i]]))] for i in chosen]


def auc_or_nan(y, s):
    # AUC pela estatistica de Mann-Whitney (empates contam meio)
    pos = int(y.sum())
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000, help="imagens por particao (uma por paciente)")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)

    with open(NIH / "Data_Entry_2017_v2020.csv", encoding="utf-8") as f:
        rows = {r["Image Index"]: r for r in csv.DictReader(f)}
    parts = {p: set((NIH / f"{p}_list.txt").read_text().split()) & set(rows) for p in ("train_val", "test")}
    images = {p: sample(parts[p], rows, args.n, rng) for p in parts}
    labels = {p: {c: np.array([c in rows[n]["Finding Labels"].split("|") for n in images[p]], dtype=int) for c in COMMON}
              for p in parts}

    tensors = {p: torch.stack([load_image(NIH / "images" / n) for n in images[p]]) for p in parts}
    scores = {}
    for w in SEEN + UNSEEN:
        model = xrv.models.DenseNet(weights=w).eval()
        idx = {c: model.pathologies.index(c) for c in COMMON}
        scores[w] = {}
        with torch.no_grad():
            for p in parts:
                out = torch.cat([model(tensors[p][i:i + 64]) for i in range(0, len(images[p]), 64)]).numpy()
                scores[w][p] = {c: out[:, idx[c]] for c in COMMON}
        print("ok", w, flush=True)

    def gaps(sel):
        # gap medio entre classes para cada modelo, com os indices de amostra dados
        res = {}
        for w in scores:
            g = []
            for c in COMMON:
                a_tr = auc_or_nan(labels["train_val"][c][sel["train_val"]], scores[w]["train_val"][c][sel["train_val"]])
                a_te = auc_or_nan(labels["test"][c][sel["test"]], scores[w]["test"][c][sel["test"]])
                g.append(a_tr - a_te)
            res[w] = float(np.nanmean(g))
        return res

    full = {p: np.arange(len(images[p])) for p in parts}
    per_class = []
    for w in scores:
        for c in COMMON:
            a_tr = auc_or_nan(labels["train_val"][c], scores[w]["train_val"][c])
            a_te = auc_or_nan(labels["test"][c], scores[w]["test"][c])
            per_class.append({"weights": w, "seen_nih": w in SEEN, "class": c, "auc_train_val": a_tr, "auc_test": a_te,
                              "gap": a_tr - a_te, "pos_train_val": int(labels["train_val"][c].sum()),
                              "pos_test": int(labels["test"][c].sum())})
    point = gaps(full)
    did = lambda g, w: g[w] - float(np.mean([g[u] for u in UNSEEN]))
    boots = {w: [] for w in SEEN}
    for _ in range(args.boot):
        sel = {p: rng.integers(len(images[p]), size=len(images[p])) for p in parts}
        g = gaps(sel)
        for w in SEEN:
            boots[w].append(did(g, w))
    summary = {
        "n_per_partition": {p: len(images[p]) for p in parts},
        "classes": COMMON,
        "mean_gap": point,
        "diff_in_diff_vs_unseen_mean": {w: {"point": did(point, w),
                                            "ci95": [float(np.percentile(boots[w], 2.5)), float(np.percentile(boots[w], 97.5))]}
                                        for w in SEEN},
        "seed": args.seed, "boot": args.boot,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "txv_leakage.json").write_text(json.dumps(summary, indent=2) + "\n")
    with open(OUT / "txv_leakage.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(per_class[0]))
        wr.writeheader()
        wr.writerows(per_class)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
