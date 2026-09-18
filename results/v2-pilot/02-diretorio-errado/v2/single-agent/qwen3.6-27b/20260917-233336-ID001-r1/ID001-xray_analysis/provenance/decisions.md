# Analysis Decisions — ID001-xray.png

## Image Context
Lateral foot and ankle radiograph. Not a chest X-ray. This immediately rules out all torchxrayvision models (trained exclusively on chest radiographs) and YOLOv8 GRAZPEDWRI-DX (trained on pediatric wrist radiographs).

---

## Analysis 1: Image Enhancement (Contrast/Histogram Equalization)
**Question**: Can enhanced contrast reveal subtle findings (fractures, cortical irregularities) not obvious in the raw image?
**Method**: CLAHE (Contrast Limited Adaptive Histogram Equalization) and gamma correction to enhance bone detail.
**Justification**: Standard radiographic preprocessing to improve visibility of low-contrast structures.
**Would change interpretation if**: A fracture line, cortical step-off, or abnormal lucency becomes clearly visible.
**Would NOT change interpretation if**: The enhanced image shows the same anatomy without new findings.
**Artifacts risk**: CLAHE can introduce blocky artifacts or amplify noise, making noise patterns look like fractures. Gamma can shift contrast misleadingly.

## Analysis 2: Bone-Fracture-Detection (siglip2-based classifier)
**Question**: Does the model detect a fracture in this foot/ankle radiograph?
**Method**: HuggingFace model `prithivMLmods/Bone-Fracture-Detection`, fine-tuned from `google/siglip2-base-patch16-224` on `Hemg/bone-fracture-detection` dataset. Labels: "Fractured" / "Not Fractured".
**Justification**: While not chest-specific, this model was trained on bone fracture detection and may generalize to foot/ankle. The training dataset (`Hemg/bone-fracture-detection`) is not well-documented for body regions.
**Would change interpretation if**: High-confidence "Fractured" output corroborates a visible finding.
**Would NOT change interpretation if**: Low-confidence output, or output contradicts visual evidence (in which case visual evidence takes precedence).
**Artifacts risk**: Model was not validated peer-reviewed. Training data may be wrist/ankle focused (from RIV-W dataset). Could give false positives or false negatives on foot laterals.
**Validation plan**: I will check whether the model's output aligns with what I see visually. If the model says "Fractured" but I don't see a fracture, I will not trust the model. If it says "Not Fractured" and I see something, I will not dismiss my observation.

## Analysis 3: BiomedCLIP Zero-Shot
**Question**: What does a general biomedical vision-language model say about this image?
**Method**: BiomedCLIP zero-shot classification with prompts relevant to foot/ankle radiographs.
**Justification**: BiomedCLIP was trained on 15M biomedical figure-caption pairs including radiographs. Zero-shot may produce useful descriptions or highlight abnormalities.
**Would change interpretation if**: The top matches describe specific abnormalities visible in the image.
**Would NOT change interpretation if**: Matches are generic or unrelated.
**Artifacts risk**: Zero-shot classification is inherently unreliable. Text prompt engineering strongly affects results. Not a diagnostic tool.

## Analysis 4: MedSAM Segmentation
**Question**: Can segmentation help delineate the bones for better structural assessment?
**Method**: MedSAM with bounding boxes around major bone groups to segment them.
**Justification**: Could help visualize bone boundaries more clearly.
**Would change interpretation if**: Segmentation boundaries reveal a discontinuity (fracture) not obvious in the raw image.
**Would NOT change interpretation if**: Segmentation simply outlines expected anatomy.
**Artifacts risk**: MedSAM may produce inaccurate segment boundaries on single-view foot X-rays (trained on diverse modalities, not specifically foot radiographs). Segmentation is not a fracture detection tool.

---

## Analysis Results

### Analysis 1: Image Enhancement
**What I saw**: CLAHE (clip=3, 8x8) provided the best improvement in bone detail and contrast. The histogram equalization introduced severe blocky artifacts that could mimic fracture lines. Gamma 0.5 made the image too dark, reducing visibility.
**Answered?**: Partially. CLAHE improved visual inspection but did not reveal any new findings beyond what was visible in the original image.
**Artifacts**: Histogram equalization produced strong artifacts. Gamma correction reduced contrast too much.
**Decision**: Keep CLAHE result for reference. Discard histogram equalization and gamma images from final output — they are not diagnostically useful and introduce artifacts.

### Analysis 2: Bone-Fracture-Detection Classifier
**What I measured**: Model predicted "Fractured" with 78.6% confidence (Fractured: 0.786, Not Fractured: 0.214).
**Answered?**: No. While the classifier flagged the image as "Fractured", my careful visual inspection of the original and CLAHE-enhanced images does not reveal a clear fracture line, cortical disruption, or step-off. This model is not peer-reviewed, was trained on an unknown dataset composition for this body region, and its output cannot be trusted to override visual findings.
**Artifacts**: Potential false positive due to training data mismatch.
**Decision**: Classifier output is NOT trusted. Final interpretation relies on visual evidence. The classifier result is documented as a negative validation finding.

### Analysis 3: BiomedCLIP Zero-Shot
**What I measured**: All prompt scores were approximately 0.05-0.06, essentially uniform. Top match was "normal lateral ankle radiograph" (0.062). The model showed no meaningful discrimination between abnormal and normal prompts.
**Answered?**: No. Results were essentially random, providing no diagnostic value.
**Artifacts**: None, but the model is simply not useful for this task.
**Decision**: Not trusted. Results documented but not used to support any finding.

### Analysis 4: MedSAM Segmentation
**What I saw**: Segmentation quality was poor. The calcaneus mask extended far below the actual bone into soft tissue. The talus mask outlined a narrow region missing most of the bone. The distal tibia fibula mask traced soft tissue rather than bone. Only the metatarsal region segmentation was approximately correct but still imprecise.
**Answered?**: No. The poor segmentation accuracy means it cannot be used for structural assessment or fracture detection.
**Artifacts**: Mask boundaries do not correspond to actual bone edges.
**Decision**: Not trusted. Discard individual mask images from final output. Keep only the composite for documentation.

## Comparison With First Look
- My initial impression was that no gross fracture or dislocation was immediately obvious. This remains unchanged after enhanced processing.
- The Bone-Fracture-Detection classifier's "Fractured" prediction (78.6%) created an apparent conflict, but visual inspection of the enhanced images confirmed that no fracture line, cortical step-off, or disruption is clearly visible.
- I did NOT change my interpretation based on the classifier output, as the model is unvalidated and visual evidence takes precedence.
- The single lateral view limitation identified in the first look remains the primary constraint.

## Discarded Analyses
- **torchxrayvision DenseNet/ResNet classifiers**: Discarded — trained on chest radiographs only.
- **PSPNet chest segmentation**: Discarded — chest-specific.
- **ChestX-anatomy UNet**: Discarded — chest-specific.
- **ViewModel classifier**: Discarded — chest view classifier.
- **DenseNet autoencoder**: Discarded — chest reconstruction.
- **JF Healthcare DenseNet**: Discarded — chest pathologies.
- **YOLOv8 GRAZPEDWRI-DX**: Discarded — trained on pediatric wrist radiographs only.
- **Histogram equalization**: Discarded — introduced severe blocky artifacts.
- **Gamma correction (0.5)**: Discarded — image too dark, reduced diagnostic value.
- **MedSAM individual masks**: Discarded from final output — poor segmentation accuracy.
- **BiomedCLIP zero-shot**: Discarded — inconclusive, near-uniform scores.
