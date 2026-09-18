# Analysis Decisions — ID001-xray.png

## Pre-analysis Model Applicability Assessment

### Models NOT used (chest radiograph models — clearly out of domain)
| Model | Reason for exclusion |
|---|---|
| xrv DenseNet (all variants) | Trained on chest radiographs (NIH, PadChest, CheXpert, MIMIC, RSNA, Google). A foot X-ray is entirely outside training distribution. |
| xrv ResNet | Same — chest radiograph classifier. |
| xrv ResNetAE (autoencoder) | Chest radiograph autoencoder. Reconstruction would be meaningless for a foot. |
| xrv PSPNet (chest structure segmentation) | Segments 14 chest structures. Not applicable. |
| xrv UNetResNet50 (chest anatomy) | Segments 159 chest anatomical structures. Not applicable. |
| xrv ViewModel | Classifies chest view (frontal vs lateral). The word "lateral" happens to match but the model is trained on chest views. |
| xrv JFHealthcare DenseNet | 5 chest pathology scores. Not applicable. |
| YOLOv8 GRAZPEDWRI-DX | Trained on pediatric wrist radiographs. While it can detect "fracture" and "boneanomaly," these were learned from wrist anatomy. May produce nonsensical bounding boxes on a foot. Will NOT use. |

### Models considered and tested
| Model | Why considered | Applicable? |
|---|---|---|
| Bone-Fracture-Detection (prithivMLmods) | Binary fracture classifier, fine-tuned from SigLIP2. The training dataset is unknown but the model name suggests bone images. Worth testing as a sanity check. | Unclear. May or may not generalize. |
| BiomedCLIP | Vision-language model trained on 15M biomedical images including radiographs from many modalities. Zero-shot classification can be tailored to foot-specific findings. | Plausible. The PMC-15M dataset includes many image types. |
| MedSAM | SAM fine-tuned on 1.5M medical image-mask pairs across 10 modalities. Can segment anatomical structures given bounding box prompts. | Plausible for segmenting bone regions. |

### Basic image processing
Standard X-ray processing techniques (contrast enhancement, edge detection) are modality-agnostic and can help visualize subtle findings.

---

## Analysis 1: Basic Image Enhancement
**Question**: Can contrast enhancement and edge detection reveal any subtle fracture lines or cortical disruptions not obvious at first glance?
**Method**: CLAHE (Contrast Limited Adaptive Histogram Equalization), bone-window contrast stretching, Sobel edge detection, and Laplacian sharpening.
**Why**: These are standard X-ray post-processing techniques that can highlight subtle cortical irregularities.
**What would change interpretation**: A clear lucent line or cortical step-off becoming visible after enhancement.
**What would not**: Enhancement showing only normal anatomy with no new findings.
**Possible artifacts**: CLAHE can create false edges at intensity transitions; edge detection highlights all intensity gradients, not just fractures.

**Result**: [to be filled after execution]

---

## Analysis 2: Bone-Fracture-Detection Model
**Question**: Does a general-purpose bone fracture classifier detect a fracture in this image?
**Method**: Run prithivMLmods/Bone-Fracture-Detection (SigLIP2-based). Check "Fractured" vs "Not Fractured" classification.
**Why**: Although the training data provenance is limited, the model was fine-tuned from a strong vision backbone (SigLIP2) and may have learned generalizable features.
**What would change interpretation**: A confident "Fractured" prediction would prompt more careful manual inspection for a fracture I may have missed.
**What would not**: A "Not Fractured" prediction or uncertain output — this model has no peer-reviewed validation for foot radiographs.
**Possible artifacts**: The model may confuse any abnormal appearance with fracture. Being out-of-domain, its output should not be trusted without corroborating evidence.
**Validation check**: I will note the confidence scores. If scores are near 0.5, the model is uncertain, which I will interpret as "inapplicable" rather than "no fracture."

**Result**: [to be filled after execution]

---

## Analysis 3: BiomedCLIP Zero-Shot Classification
**Question**: What radiological findings does a biomedical vision-language model associate with this image?
**Method**: Use BiomedCLIP with a set of candidate labels specific to foot/ankle radiographs, plus negative controls.
**Candidate labels**: "normal foot radiograph", "fracture of the foot", "fracture of the ankle", "calcaneal fracture", "metatarsal fracture", "talar fracture", "osteoarthritis of the ankle", "subtalar arthritis", "soft tissue swelling", "foreign body in foot", "bone tumor of foot"
**Why**: BiomedCLIP was trained on a diverse set of biomedical images and may generalize to foot radiographs better than models trained only on chest X-rays.
**What would change interpretation**: A clear top-1 prediction of "fracture" or other pathology with high score separation from "normal."
**What would not**: Uniformly low scores across all categories (indicating out-of-domain) or "normal" as top-1 with other categories also low.
**Validation check**: I will compare the ranking to my manual reading. If BiomedCLIP says "normal" and I see normal, that adds confidence. If it says "fracture" and I don't see one, I will look harder but not accept it without visual evidence.

**Result**: [to be filled after execution]

---

## Analysis 4: MedSAM Segmentation
**Question**: Can I segment the bone structures of the foot to inspect their shape and cortical margins more carefully?
**Method**: Provide bounding box prompts around major bone regions (distal tibia, talus, calcaneus, midfoot) and use MedSAM to generate masks.
**Why**: If MedSAM successfully segments bone, the masks can be overlaid on the original to verify that no cortical irregularities are being overlooked.
**What would change interpretation**: A mask that clearly shows a discontinuity in bone cortex (though this is unlikely — MedSAM segments regions, not detects edges).
**What would not**: Masks that look like normal bone outlines.
**Possible artifacts**: MedSAM was fine-tuned on diverse medical images but may not handle X-ray bone segmentation well; masks may be incomplete or include soft tissue.

**Result**: [to be filled after execution]
