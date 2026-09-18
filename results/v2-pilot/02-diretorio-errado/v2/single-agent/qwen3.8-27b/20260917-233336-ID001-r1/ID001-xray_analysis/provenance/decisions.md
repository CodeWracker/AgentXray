# Analysis decisions

Chronological record of every analysis decided, run, evaluated, and discarded.

## A. Image enhancement to reveal subtle fracture lines

- **Question:** Are there subtle cortical disruptions (hairline fractures, cortical steps) invisible in the original at 373×454 px?
- **Method:** (a) global histogram analysis, (b) CLAHE local contrast enhancement, (c) contrast stretch (percentile window), (d) unsharp/high-pass mask. Standard radiology display tricks; no model involved.
- **Why it may answer:** enhancing local contrast can make thin cortices and subtle lucent lines stand out against noise; high-pass emphasizes edges.
- **What would change interpretation:** a clear lucent line crossing a cortex with associated cortical step or bone fragment → new fracture finding. No new structure, just clearer cortex → confirms "no obvious fracture" at this resolution.
- **Artifacts:** CLAHE/high-pass can create false "lines" at tissue-bone boundaries and noise; a bright edge at a soft-tissue contour is not a fracture. Must compare against original before believing anything.

## B. BiomedCLIP zero-shot (vision-language)

- **Question:** (1) Does an independent model agree this is a lateral foot/leg radiograph? (2) Does zero-shot similarity give any signal about "fracture" vs "normal"?
- **Why it may answer:** BiomedCLIP is trained on 15M PMC figure–caption pairs including radiographs of many body regions, so body-region/projection zero-shot classification is within its training distribution. It is the only tool in the inventory that is genuinely body-agnostic.
- **What would change interpretation:** a confident "chest" or "hand" classification would contradict my visual read and force re-examination (unlikely). A fracture-vs-normal prompt is a weak zero-shot probe; only a *decisive* fracture score would make me look harder at the image, and even then I would verify visually first.
- **How I will check it is sensible:** run it on this image with a body-region prompt set that includes the correct answer and clear distractors; expect foot/leg to win with high cosine similarity. If all scores are near-flat or it picks something implausible (e.g., "MRI of brain"), I will not trust any of its outputs.
- **Artifacts:** zero-shot single-word prompts are unstable; "fracture" prompts may match text about *detecting* fractures in captions rather than actual fracture appearance. Documented as low-trust for fracture detection, usable for body-region sanity.

## C. Bone-Fracture-Detection classifier (SigLIP2, `prithivMLmods/Bone-Fracture-Detection`)

- **Question:** Does a dedicated fracture binary classifier detect a fracture on this foot?
- **Why it may answer:** it is trained specifically for fracture vs not-fracture on the HF dataset `Hemg/bone-fracture-detection`.
- **In-domain check:** the model card does **not** state which body regions the training set covers (dataset name is generic). I cannot verify a foot radiograph is in its training distribution from local files. I will (1) run it on this image, (2) run it on a leg-only crop and a foot-only crop to check response stability, and (3) treat the result as *supportive at best*, never as primary evidence.
- **What would change interpretation:** high-confidence "Fractured" → I would search the image harder (likely at metatarsals/phalanges) before believing it. High-confidence "Not Fractured" → consistent with my visual read, supportive only.
- **Artifacts:** binary softmax on an out-of-distribution body region can be confidently wrong; OOD images can sit anywhere on the decision boundary.

## D. YOLOv8 GRAZPEDWRI-DX detector

- **Question:** Does a fracture/trauma detector produce any detections here?
- **In-domain check:** trained on ~20k **pediatric wrist** trauma radiographs (GRAZPEDWRI-DX). A foot/leg radiograph is clearly **out of distribution** for this model. I will run it once to document its behavior on OOD input; I do not expect its outputs to be meaningful and will not let them support a finding.
- **What would change interpretation:** nothing by itself; a "fracture" detection on a foot image from a wrist-trained model would be treated as a false positive to document, not a finding.

## E. torchxrayvision chest classifiers (DenseNet-121 all-dataset, ResNet-50)

- **Question:** (negative control) how do chest-X-ray pathology models behave on a foot radiograph?
- **Why:** every torchxrayvision classifier in the inventory is trained on **chest** radiographs (NIH/PadChest/CheXpert/MIMIC/RSNA/SIIM/VinDr). This image is a foot, so these are out of distribution. I will run DenseNet-121 (224) once to document the OOD behavior and confirm I should **not** use any chest model for findings on this image.
- **What would change interpretation:** nothing — this is a documented rejection. If any pathology score were high, it would reflect OOD nonsense, not a chest finding (there is no chest in this image).

## F. Geometric alignment measurements (model-free)

- **Question:** Are the ankle and midfoot congruent — i.e., no dislocation/subluxation?
- **Method:** manual landmark picking is not reproducible from a script, so instead I will measure objective image statistics: (1) intensity profile across the tibiotalar joint (joint-space darkness band), (2) orientation angles of the tibia, talus and calcaneus from thresholded bone masks (PCA on component masks), (3) check the talus remains under the tibial plafond (centroid alignment). A dislocation would grossly perturb these.
- **What would change interpretation:** gross misalignment of tibia–talus or talus–calcaneus axes → dislocation/subluxation to verify visually. Consistent axes → supports "alignment preserved".
- **Artifacts:** thresholding at low resolution merges adjacent bones; PCA angles on merged masks are only coarse. This is a coarse screen, not a measurement of joint congruence in the clinical sense.

## G. MedSAM promptable segmentation (candidate)

- **Question:** Can I obtain bone-only masks (calcaneus, talus, metatarsals) to inspect cortical continuity on isolated bones?
- **Why it may answer:** MedSAM (SAM fine-tuned on 1.5M medical masks, 10 modalities including X-ray) with a tight box around a single bone may yield a clean mask; overlaying it on the original highlights the bone cortex.
- **What would change interpretation:** a mask that follows a cortex with a clear step/discontinuity → fracture evidence to verify in the original.
- **Artifacts:** SAM-style models often leak across adjacent bones or include soft tissue at box edges; a jagged mask edge is a segmentation artifact, not a fracture. I will run this only if A–F leave a specific unresolved region.

## H. torchxrayvision autoencoder (ResNetAE 101-elastic) — candidate negative control

- **Question:** does reconstruction error highlight anything anomalous?
- **Why it may answer:** an autoencoder trained on **chest** X-rays reconstructs "typical" images; large reconstruction error could in principle flag unusual structure. But the training domain is chest, so a foot will be globally "unusual" — error maps will light up on the whole foot. Expected to be **useless** here; I will run it only if I need another documented negative result, or skip. Decision: run once, cheap, and document.

---

## Result A (enhancement)

- **What I saw/measured:** Histogram: image content is dark (bone peaks ~100-150, background 0-10, p2=11, p98=149). CLAHE (clip 2.0, 8×8) produced a clean, well-balanced view; unsharp (sigma 3, amount 2) was similar. Inspected 4× zooms of ankle, midfoot/tarsus and forefoot on the CLAHE image: cortical outlines of distal tibia, fibula, talus, calcaneus, navicular, cuneiforms and metatarsals appear **continuous**; tibiotalar joint congruent; no lucent line crossing any cortex, no cortical step, no displaced fragment.
- **Answered the question:** Partially. No *obvious or moderate* fracture is visible after enhancement. A hairline fracture below this image's resolution cannot be excluded — that remains an inherent limitation, not a failed test.
- **Artifacts:** The 2–98% stretch version (02) and stretch+CLAHE (04) overexposed the bones to saturated white and lost all trabecular/cortical detail — useless. **Discarded** (deleted from images/). CLAHE and unsharp introduced no misleading "fracture-like" lines.
- **Next:** body-region/projection sanity check with BiomedCLIP (B), then fracture-specific models (C, D) and chest-model negative controls (E).

## Result B (BiomedCLIP zero-shot)

- **Sensibility check (how I validated it):** ran body-region prompt set containing the correct answer (foot/ankle) plus distractors (hand, wrist, chest, knee). The model behaved sensibly: "lateral radiograph of the foot and ankle" = 0.485, "radiograph of the foot" = 0.451, "lower leg and foot" = 0.447, vs chest 0.255 / hand 0.284 / wrist 0.321 / knee 0.326. Softmax (T=50) over the set: 44.5% foot-and-ankle-lateral, 22.6% foot, 20.6% lower-leg+foot, <5% each distractor. Clear, plausible margin → model is functioning on this image type.
- **Conclusion:** independent confirmation that this is a **lateral radiograph of the foot/ankle (lower leg)**, matching my visual read. Projection prompts: "lateral" (0.369) slightly above "AP" (0.353) — weak, inconclusive on projection, but visual read already established lateral with certainty.
- **Fracture probe (low trust, by design):** "intact foot" 0.449 and "normal without fracture" 0.412 outscore "fracture of the bone" 0.367 and "broken bone" 0.325. Direction is consistent with no obvious fracture, but single-word zero-shot probes are unstable; **not** used as evidence for a finding, only as a weak corroborating signal.
- **Answered the question:** yes for body region/projection sanity (trust: true, for body region only). Fracture signal: no.
- **Next:** C (dedicated fracture classifier), D (wrist-trained detector as OOD probe), E (chest classifier negative control).

## Result C (Bone-Fracture-Detection, SigLIP2)

- **What I measured:** full image → Fractured 0.786 / Not Fractured 0.214. Leg-only crop → Fractured 0.572. Foot-only crop → Fractured 0.994. **Sanity control: uniform gray image (no anatomy) → Fractured 0.9985.**
- **In-domain check / trust decision:** the model card does not document which body regions are in its training set (`Hemg/bone-fracture-detection`), so a foot is not a confirmed in-domain input. More decisively, the no-anatomy sanity control is classified "Fractured" at 99.85% confidence — a well-calibrated fracture classifier must not be maximally confident on a blank image. The model is effectively responding to global style/OOD, not to fracture evidence. **trusted: false.** Its "Fractured" output on this foot is a **documented negative result** and does NOT support a fracture finding.
- **Answered the question:** no — the model is untrustworthy on this image, so it neither confirms nor refutes a fracture.
- **Next:** D (wrist-trained detector OOD probe) and E (chest classifier negative control) to complete the "model outputs do not drive this read" documentation.

## Result D (YOLOv8 GRAZPEDWRI-DX)

- **What I measured:** 0 detections on the full image (all 9 wrist-classes).
- **Trust decision:** model is trained exclusively on pediatric **wrist** trauma radiographs; a foot/leg is clearly out of distribution. No output is meaningful here. **trusted: false.** Documented as an OOD negative result only.
- **Answered the question:** no (by design).

## Result E (DenseNet-121 chest pathology classifier) and H (ResNetAE 101-elastic)

- **What I measured:**
  - DenseNet-121 (224, all-datasets) returned 18 **chest** pathology scores on a foot image: e.g. Atelectasis 0.50, Edema 0.52, Pneumonia 0.51, Emphysema 0.51, "Fracture" (rib-fracture label) 0.53, Cardiomegaly 0.43, etc.
  - Autoencoder reconstruction error (224): global_mean 35.5, p95 111, max 1295 (input range is [-1024,1024]). The saved error map (05_autoencoder_error.png) lights up **every bone edge uniformly** — the model simply cannot reconstruct a foot.
- **In-domain check / trust decision:** both are trained exclusively on **chest** radiographs (NIH/PadChest/CheXpert/MIMIC/RSNA). A foot/leg is maximally out of distribution. The DenseNet "Fracture" score of 0.53 refers to a *chest rib fracture* label and says nothing about a foot; the AE error map is a global OOD failure, not a lesion. **trusted: false** for both. Kept only as documented negative controls proving that *no* chest-trained model in the inventory can support a finding on this image.
- **Answered the question:** no (by design) — confirms my read must rest on direct image inspection plus the two foot-agnostic/limb tools (BiomedCLIP body-region, and any limb fracture tool that passes a sanity check).
- **Next:** F (model-free geometric alignment screen) and G (MedSAM bone isolation) to back the "no dislocation / no obvious fracture" claim with non-chest evidence.
