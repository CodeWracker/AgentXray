#!/usr/bin/env python3
"""Catalogo ontologico clinico para mapeamento de termos livres para classes canonicadas.

Mapeia descricoes radiologicas, achados e diagnosticos diferenciais em texto livre
para os conceitos padronizados do NIH ChestX-ray14 usando correspondencias do RadLex,
SNOMED CT e MeSH.

Uso:
    from clinical_ontology import ONTOLOGY, project_hypothesis, match_term_to_class, LEXICON_VERSION
"""

import hashlib
import json
import re

# Mapeamento canônico das 14 patologias + Normal
ONTOLOGY = {
    "Atelectasis": {
        "snomed_id": "46621007",
        "radlex_id": "RID4952",
        "mesh_id": "D001299",
        "synonyms": [
            "atelectasis",
            "atelectatic",
            "lung collapse",
            "collapsed lung segment",
            "lobar collapse",
            "subsegmental collapse",
            "volume loss",
            "discoidal atelectasis",
            "plate-like atelectasis",
            "linear atelectasis",
            "absorptive atelectasis",
            "compressive atelectasis",
        ],
    },
    "Cardiomegaly": {
        "snomed_id": "8186001",
        "radlex_id": "RID5291",
        "mesh_id": "D006332",
        "synonyms": [
            "cardiomegaly",
            "enlarged heart",
            "cardiac enlargement",
            "increased cardiothoracic ratio",
            "enlarged cardiac silhouette",
            "prominent cardiac contour",
            "biventricular enlargement",
            "left ventricular enlargement",
            "right ventricular enlargement",
        ],
    },
    "Effusion": {
        "snomed_id": "60046008",
        "radlex_id": "RID5361",
        "mesh_id": "D010996",
        "synonyms": [
            "effusion",
            "pleural effusion",
            "pleural fluid",
            "fluid in pleural space",
            "blunting of costophrenic angle",
            "blunted costophrenic sulcus",
            "blunting of CP angle",
            "fluid collection in pleura",
            "hemothorax",
            "hydrothorax",
            "meniscus sign",
        ],
    },
    "Infiltration": {
        "snomed_id": "46866001",
        "radlex_id": "RID5378",
        "mesh_id": "D011654",
        "synonyms": [
            "infiltration",
            "infiltrate",
            "infiltrates",
            "pulmonary infiltrate",
            "patchy infiltrate",
            "interstitial infiltrate",
            "airspace disease",
            "opacification",
            "patchy opacity",
            "reticulonodular opacity",
            "ill-defined opacity",
            "hazy opacity",
        ],
    },
    "Mass": {
        "snomed_id": "4147007",
        "radlex_id": "RID3875",
        "mesh_id": "D008175",
        "synonyms": [
            "mass",
            "pulmonary mass",
            "lung mass",
            "pleural mass",
            "mediastinal mass",
            "large lesion",
            "neoplasm",
            "space-occupying lesion",
            "solitary mass",
            "lesion > 3cm",
        ],
    },
    "Nodule": {
        "snomed_id": "27925004",
        "radlex_id": "RID3874",
        "mesh_id": "D002933",
        "synonyms": [
            "nodule",
            "nodules",
            "pulmonary nodule",
            "lung nodule",
            "solitary pulmonary nodule",
            "coin lesion",
            "granuloma",
            "micronodule",
            "subcentimeter nodule",
            "calcified nodule",
            "focal round opacity",
        ],
    },
    "Pneumonia": {
        "snomed_id": "233604007",
        "radlex_id": "RID5357",
        "mesh_id": "D011014",
        "synonyms": [
            "pneumonia",
            "bronchopneumonia",
            "lobar pneumonia",
            "infectious consolidation",
            "lung infection",
            "pneumonic infiltrate",
            "community-acquired pneumonia",
            "aspiration pneumonia",
            "atypical pneumonia",
        ],
    },
    "Pneumothorax": {
        "snomed_id": "36118008",
        "radlex_id": "RID5340",
        "mesh_id": "D011129",
        "synonyms": [
            "pneumothorax",
            "pleural air",
            "air in pleural cavity",
            "air in pleural space",
            "apical pneumothorax",
            "tension pneumothorax",
            "visceral pleural line",
            "absence of vascular markings",
            "deep sulcus sign",
        ],
    },
    "Consolidation": {
        "snomed_id": "75570004",
        "radlex_id": "RID5356",
        "mesh_id": "D000081017",
        "synonyms": [
            "consolidation",
            "lobar consolidation",
            "dense parenchymal opacity",
            "alveolar consolidation",
            "air bronchogram",
            "confluent opacity",
            "dense pulmonary opacity",
        ],
    },
    "Edema": {
        "snomed_id": "19242006",
        "radlex_id": "RID5360",
        "mesh_id": "D011654",
        "synonyms": [
            "edema",
            "pulmonary edema",
            "interstitial edema",
            "alveolar edema",
            "cardiogenic edema",
            "kerley b lines",
            "bat wing opacity",
            "vascular congestion",
            "fluid overload",
            "peribronchial cuffing",
        ],
    },
    "Emphysema": {
        "snomed_id": "87433001",
        "radlex_id": "RID5388",
        "mesh_id": "D011656",
        "synonyms": [
            "emphysema",
            "hyperinflation",
            "pulmonary emphysema",
            "bullous emphysema",
            "flattened diaphragms",
            "increased retrosternal space",
            "barrelling of chest",
            "centrilobular emphysema",
        ],
    },
    "Fibrosis": {
        "snomed_id": "51615001",
        "radlex_id": "RID5367",
        "mesh_id": "D011658",
        "synonyms": [
            "fibrosis",
            "pulmonary fibrosis",
            "interstitial fibrosis",
            "reticular markings",
            "reticular pattern",
            "honeycombing",
            "traction bronchiectasis",
            "apical scarring",
            "parenchymal scarring",
        ],
    },
    "Pleural_Thickening": {
        "snomed_id": "116244005",
        "radlex_id": "RID5365",
        "mesh_id": "D010998",
        "synonyms": [
            "pleural thickening",
            "pleural plaque",
            "calcified pleural plaque",
            "apical capping",
            "pleural scarring",
            "blunting secondary to pleural thickening",
            "diffuse pleural thickening",
        ],
    },
    "Hernia": {
        "snomed_id": "52515009",
        "radlex_id": "RID5394",
        "mesh_id": "D006547",
        "synonyms": [
            "hernia",
            "hiatal hernia",
            "diaphragmatic hernia",
            "retrocardiac mass with air-fluid level",
            "retrocardiac lucency",
            "intrathoracic stomach",
            "hiatus hernia",
        ],
    },
    "No Finding": {
        "snomed_id": "17621005",
        "radlex_id": "RID13173",
        "mesh_id": "D006262",
        "synonyms": [
            "normal",
            "no finding",
            "no findings",
            "no acute finding",
            "no acute cardiopulmonary disease",
            "clear lungs",
            "unremarkable",
            "within normal limits",
            "no evidence of acute disease",
            "negative chest",
            "normal chest radiograph",
        ],
    },
}


MATCHER_VERSION = "3"

# gatilhos no esquema do NegEx (Chapman et al., 2001), com janela de ate NEG_WINDOW tokens dentro da clausula
PRE_NEGATION = ["no evidence of", "no signs of", "no sign of", "negative for", "absence of", "free of",
                "without", "no", "not"]
POST_NEGATION = ["ruled out", "excluded", "is absent", "are absent", "not seen", "not present",
                 "not identified", "has resolved", "resolved"]
# expressoes regulares de frases que contem gatilhos mas nao negam (numa lista diferencial,
# "cannot be excluded" e "with or without" mantem o candidato afirmado)
PSEUDO_NEGATION = [r"(?:can ?not|can't|not) be (?:\w+ )?(?:excluded|ruled out)",
                   r"not (?:\w+ )?(?:excluded|ruled out)", r"with or without", r"not only",
                   r"no (?:significant |interval )?change"]
NEG_WINDOW = 6
CLAUSE_BREAK = re.compile(r"[;,.:()\[\]]|\b(?:but|however|although|though|whereas|versus|vs|aside from|apart from"
                          r"|except(?: for)?|other than)\b")
NON_CONTENT = {"", "n/a", "na", "none", "null", "nil", "not applicable", "unknown", "-"}

LEXICON_VERSION = hashlib.sha256(json.dumps(
    {"ontology": ONTOLOGY, "matcher": MATCHER_VERSION, "pre": PRE_NEGATION, "post": POST_NEGATION,
     "pseudo": PSEUDO_NEGATION, "window": NEG_WINDOW, "non_content": sorted(NON_CONTENT)},
    sort_keys=True).encode()).hexdigest()[:12]


def _term_regex(term: str) -> re.Pattern:
    # fronteira de palavra com plural opcional: "infiltrate" casa "infiltrates", "mass" nao casa "massive"
    return re.compile(r"(?<![a-z0-9])" + re.escape(term) + r"(?:e?s)?(?![a-z0-9])")


_SYNONYM_PATTERNS = [(cls, _term_regex(syn)) for cls, info in ONTOLOGY.items() for syn in info["synonyms"]]
def _exact_regex(term: str) -> re.Pattern:
    # gatilhos sao expressoes fixas: sem o plural opcional dos sinonimos ("not" nao pode casar "notes")
    return re.compile(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])")


_PRE = [_exact_regex(t) for t in PRE_NEGATION]
_POST = [_exact_regex(t) for t in POST_NEGATION]
# "no finding of X" e "no findings suggestive of X" negam X; nao afirmam exame normal
_NOT_NORMALITY = re.compile(r"\s+(?:of|to suggest|suggest\w*|for|consistent with|related to)\b")
_PSEUDO = [re.compile(r"(?<![a-z0-9])" + t + r"(?![a-z0-9])") for t in PSEUDO_NEGATION]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip().strip(" .;:-")


def _clause_bounds(text: str) -> list[tuple[int, int]]:
    bounds, start = [], 0
    for m in CLAUSE_BREAK.finditer(text):
        bounds.append((start, m.start()))
        start = m.end()
    bounds.append((start, len(text)))
    return bounds


def _tokens_between(text: str, a: int, b: int) -> int:
    return len(re.findall(r"[a-z0-9]+", text[a:b]))


def project_hypothesis(text: str) -> dict:
    """Projeta uma hipotese em texto livre nas classes canonicas.

    Retorna {"informative", "classes" (afirmadas, em ordem de aparicao), "negated" (so negadas)}.
    """
    t = normalize_text(text)
    if t in NON_CONTENT:
        return {"informative": False, "classes": [], "negated": []}

    # mascara as pseudo-negacoes para que seus gatilhos internos nao contem
    masked = t
    for pat in _PSEUDO:
        masked = pat.sub(lambda m: " " * len(m.group(0)), masked)

    found = []
    for cls, pat in _SYNONYM_PATTERNS:
        for m in pat.finditer(t):
            if cls == "No Finding" and m.group(0).startswith("no ") and _NOT_NORMALITY.match(t, m.end()):
                continue
            found.append((m.start(), m.end(), cls))
    # a correspondencia mais longa vence quando uma esta contida em outra de classe diferente
    found.sort(key=lambda x: (-(x[1] - x[0]), x[0]))
    kept = []
    for s, e, cls in found:
        if any(ks <= s and e <= ke and kc != cls for ks, ke, kc in kept):
            continue
        kept.append((s, e, cls))

    clauses = _clause_bounds(masked)
    triggers_pre = [(m.start(), m.end()) for pat in _PRE for m in pat.finditer(masked)]
    triggers_post = [(m.start(), m.end()) for pat in _POST for m in pat.finditer(masked)]

    status: dict[str, bool] = {}
    order: dict[str, int] = {}
    for s, e, cls in kept:
        cs, ce = next(((a, b) for a, b in clauses if a <= s < b), (0, len(t)))
        negated = False
        for ts, te in triggers_pre:
            # gatilhos dentro do proprio sinonimo ("absence of vascular markings", "no finding") nao negam
            if cs <= ts and te <= s and not (s <= ts < e) and _tokens_between(masked, te, s) <= NEG_WINDOW:
                negated = True
        for ts, te in triggers_post:
            if e <= ts and te <= ce and _tokens_between(masked, e, ts) <= NEG_WINDOW:
                negated = True
        status[cls] = status.get(cls, False) or not negated
        order[cls] = min(order.get(cls, s), s)

    ranked = sorted(status, key=lambda c: order[c])
    return {
        "informative": True,
        "classes": [c for c in ranked if status[c]],
        "negated": [c for c in ranked if not status[c]],
    }


def match_term_to_class(text: str) -> list[str]:
    """Mapeia uma string de diagnostico ou hipotese para as classes canonicas afirmadas."""
    return project_hypothesis(text)["classes"]
