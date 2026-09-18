#!/usr/bin/env python3
"""Catalogo ontologico clinico para mapeamento de termos livres para classes canonicadas.

Mapeia descricoes radiologicas, achados e diagnosticos diferenciais em texto livre
para os conceitos padronizados do NIH ChestX-ray14 usando correspondencias do RadLex,
SNOMED CT e MeSH.

Uso:
    from clinical_ontology import ONTOLOGY, match_term_to_class
"""

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


def match_term_to_class(text: str) -> list[str]:
    """Mapeia uma string de diagnostico ou hipotese para classes canonicas."""
    text_lower = text.lower()
    matches = []
    for cls_name, info in ONTOLOGY.items():
        for syn in info["synonyms"]:
            if syn in text_lower:
                matches.append(cls_name)
                break
    return matches
