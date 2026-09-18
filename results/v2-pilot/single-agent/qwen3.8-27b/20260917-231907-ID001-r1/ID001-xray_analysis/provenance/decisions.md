# Decisions and evaluation log

## Planning

Key constraint established from the first look: the image is a **lateral radiograph of the
lower leg/ankle/foot**. This is out of domain for every chest-trained model in the inventory
(DenseNet/ResNet pathology classifiers, PSPNet, UNetResNet50, ViewModel, JFHealthcare) and
for the pediatric-wrist YOLO model. The potentially relevant tools are therefore:

- Classical display/edge processing (my own vision is the primary interpreter).
- BiomedCLIP zero-shot (general biomedical image-text model; radiographs of many body
  regions are in its training distribution PMC-15M).
- Bone-Fracture-Detection classifier (SigLIP2 fine-tuned on a general bone-X-ray fracture
  dataset; the closest in-domain classifier available).
- MedSAM (general promptable segmentation, X-ray among its 10 modalities).
- Chest models and wrist YOLO: run once each as documented out-of-domain negative checks.

---

## Analysis A — Image characterization (numeric baseline)

- **Question**: What are the intensity/display properties and the effective detail limit of
  this image?
- **Why this method**: Percentile statistics, histogram and a Laplacian-variance sharpness
  measure establish the display window to use for enhancement and quantify how close we are
  to the sampling limit, before any visual interpretation of processed images.
- **Would change interpretation**: If the image is essentially a 2-3 gray-level silhouette
  (very low information), findings below a few pixels cannot be trusted and "no definite
  abnormality" becomes the ceiling of what can be stated. A normal radiograph-like dynamic
  range would not change interpretation by itself.
- **Artifacts/limits**: None (pure measurement).

## Analysis B — Display enhancement + magnified ROI views

- **Question**: Are there cortical breaks, joint-space changes, lesions, soft-tissue
  swelling or cropping of the toes that are visible only at higher contrast/magnification?
- **Why this method**: Contrast-limited adaptive histogram equalization (CLAHE) plus a
  percentile stretch, and 3x upsampled crops of the ankle, calcaneus/talus, midfoot,
  forefoot/toes and distal tibia, maximize what my own vision can resolve. This is the core
  of the read; model outputs only support it.
- **Would change interpretation**: Any visible cortical discontinuity, lucency, joint-space
  narrowing or soft-tissue mass on these views would become a (tentative) finding; clean
  views would support "no definite abnormality".
- **Artifacts/limits**: CLAHE can amplify noise into blotchy texture that mimics a lesion;
  upscaling adds no information. Every processed view must be checked against the original.

## Analysis C — Line/edge enhancement (Frangi black-ridge + Canny)

- **Question**: Is there a thin dark fracture line through a bone (tibia, talus, calcaneus,
  metatarsals) that is invisible at normal contrast?
- **Why this method**: A dark fracture line is a thin dark ridge on a bright (bone)
  background; the Frangi filter with black-ridge mode enhances exactly that class of
  structure. Canny edges give a cortical contour map to check continuity.
- **Would change interpretation**: A coherent dark line traversing a bone cortex on the
  Frangi image, also visible (faintly) on the CLAHE/original image, would upgrade a finding
  to "suspected fracture". Nothing visible, or only scattered noise specks, would not change
  interpretation.
- **Artifacts/limits**: Frangi on noisy 8-bit data highlights noise streaks and tissue
  interfaces (e.g., soft-tissue borders) as false lines; Canny is edge-agnostic and will
  light up every bone edge. Must never be used alone as evidence.

## Analysis D — BiomedCLIP zero-shot

- **Question**: (a) Does a general biomedical image model recognize this as a foot/ankle
  radiograph (in-domain sanity check)? (b) Does it lean toward "fracture" vs "normal" for
  this image?
- **Why this method**: BiomedCLIP (ViT-B/16 on PMC-15M figure-caption pairs) has seen
  radiographs of many body regions, so it is the only vision-language model here whose
  training distribution plausibly includes foot radiographs. Zero-shot prompt comparison is
  its intended use.
- **How in-domain-ness is checked**: run view/region discrimination prompts
  (foot/ankle vs chest vs wrist vs knee vs lower leg). If the top matches are foot/ankle/
  lower-leg terms, the image is within its experience and its fracture-vs-normal comparison
  can be used as weak supporting evidence. If it confuses this with a chest or wrist image,
  its outputs are discarded.
- **Would change interpretation**: A consistent (across several prompt phrasings) higher
  similarity to "fracture" prompts than to "normal" prompts would be a yellow flag to re-examine
  the ROI views harder; the reverse supports a normal read. It cannot by itself create a
  finding (no localization, weakly calibrated).
- **Artifacts/limits**: Prompt-wording sensitivity, no calibration of raw cosine scores,
  no localization. Treated as supporting evidence only.

## Analysis E — Bone-Fracture-Detection classifier

- **Question**: Does the general bone-X-ray fracture classifier flag this image?
- **Why this method**: It is the only classifier in the inventory trained directly on the
  question "fracture or not" on bone radiographs (community model, SigLIP2-based, reported
  ~83% test accuracy on its own dataset).
- **How in-domain-ness / sensible behavior is checked before trusting it**:
  1. negative control: run on a blank (all-black) image — a sensible model should not be
     confidently "fractured" on pure noise/blank;
  2. consistency: run on the full image and on crops that clearly contain bone (ankle,
     heel) vs a crop that is mostly soft tissue; wild, unstable swings between clearly
     similar bone crops would indicate out-of-domain confusion;
  3. raw logits per class are recorded (not just argmax) to see the margin.
- **Would change interpretation**: A high "Fractured" probability (clear margin) would be a
  reason to scrutinize the ROI views for a specific fracture site (though it cannot say
  where). A low or borderline probability supports the normal read. Either way it is only
  supporting evidence.
- **Artifacts/limits**: Community model without peer-reviewed validation; training dataset
  composition unknown to me; global decision with no localization; input resized to 224
  (loses detail).

## Analysis F — MedSAM box-prompted segmentation

- **Question**: Where exactly are the calcaneus, talus/ankle and distal tibia, and do their
  segmented boundaries look anatomically plausible?
- **Why this method**: MedSAM (SAM ViT-B fine-tuned on ~1.5M medical image-mask pairs across
  10 modalities including radiographs) is in domain as a general medical segmenter; box
  prompts give it the target. The overlays let me confirm anatomy and frame the cortical
  margins precisely for visual inspection.
- **Would change interpretation**: If MedSAM's boxes track the bones cleanly, I gain
  confidence in my own anatomical labels and in which pixels belong to which bone when I
  look for cortical breaks. Garbage masks (tracking background/soft tissue) would mean I
  discard the tool and rely on B/C alone.
- **Artifacts/limits**: Promptable segmentation is not fracture-aware; masks can leak into
  adjacent bone or soft tissue; used for localization only, never as evidence of pathology.

## Analysis G — Out-of-domain model negative checks (documented)

- **Question**: Do the chest-trained models and the pediatric-wrist YOLO behave sensibly on
  this image? (Expectation: no — to document why they are not used for findings.)
- **Why this method**: A one-pass run of xrv DenseNet121 (18 chest pathology scores), xrv
  ViewModel (chest frontal/lateral logits) and YOLOv8 GRAZPEDWRI-DX (wrist trauma detector)
  on a foot image. Nonsense outputs confirm out-of-domain and justify exclusion.
- **Would change interpretation**: If any of these produced a confident, anatomically
  plausible-looking output for a foot/ankle structure, I would have to reconsider its
  applicability. Expected: low-entropy confusion, high-entropy pathology scores, or
  empty/garbage detections — in which case nothing changes and they remain excluded.
- **Artifacts/limits**: These outputs are never used as evidence; they are recorded as
  documented negative results only.

## Sequencing

A → B → C (classical, vision-driven) first, because they determine the read. D, E, F, G
(model-based) run afterward to cross-check, with their results only able to *modify* (add or
qualify) findings that are visible in the original/enhanced images — never to create one.

---

## Evaluation log (appended after each run)
