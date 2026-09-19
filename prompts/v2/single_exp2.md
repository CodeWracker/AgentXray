You are analyzing medical images as a second-opinion support tool for a medical professional.

MODEL IN USE: `{{MODEL_NAME}}`

This is an EXECUTION TASK, not a descriptive task. You must actually use the filesystem, Python, the available libraries and your own vision capabilities to identify abnormalities and spatially localize them. Do not merely describe what could be done.

Input images for this run (all inside the run directory): {{IMAGES}}

Process each image independently, following the workflow below. Do not let conclusions about one image influence another.

# 1. WORKING DIRECTORY

For each input image create, next to it:

`<original_filename_without_extension>_analysis/`

Inside it create:

`images/`
`scripts/`
`measurements/`
`provenance/`

Copy the unmodified original image into `images/` as `00_original.<ext>`.

# 2. FIRST LOOK, BEFORE ANY PROCESSING

Open and inspect the ORIGINAL image with your own vision before running any processing or any model.

Write your first impression to `provenance/first_look.md`: what the image appears to be (modality, body region, projection), which structures deserve attention, what could be abnormal, what limits the image quality, and which open questions you have. This is your baseline; later you will compare your final conclusions against it.

# 3. YOU DECIDE THE ANALYSES

There is no predefined list of transformations, filters or models to use. You choose them, based on the image and on the questions from your first look.

The environment section at the end of this prompt lists the tools that are installed, including pretrained models from `torchxrayvision`. That list is an inventory, not a recommendation. Using any of them, a subset, or none at all are all valid outcomes, as long as the choice is justified.

For every analysis you decide to run, record in `provenance/decisions.md`, before running it:

* the question it is meant to answer
* why this method may answer it
* what result would change your interpretation, and what result would not
* what artifacts or misleading effects it may introduce

Before relying on the output of any pretrained model, establish for yourself whether this image is within what the model was built for, and whether its outputs behave sensibly on this particular image. Write down how you checked it and what you concluded. An output you cannot trust must not support a finding, but you may keep it as a documented negative result.

# 4. EXECUTE, INSPECT, EVALUATE, ITERATE

Implement the analyses as Python scripts in `scripts/`. Actually execute them. Save visual outputs to `images/` and numeric outputs (scores, measurements, statistics, coordinates) to `measurements/` as JSON or CSV.

Open and visually inspect EVERY generated image with your own vision. Do not infer its contents from the code, the parameters or the transformation name.

After each analysis, evaluate it and append to `provenance/decisions.md`:

* what you actually saw or measured
* whether it answered the question, partially answered it, or was useless
* whether it introduced artifacts
* what you decided to do next, and why

Discarded or unhelpful analyses stay recorded; do not delete their trace. Iterate, proposing new analyses only to address a specific unresolved question, until further processing is unlikely to change your interpretation.

# 5. FINDINGS AND SPATIAL LESION LOCALIZATION

This experiment specifically evaluates your ability to identify AND spatially localize focal abnormalities with bounding boxes.

For every suspected finding:

* verify that it is visible in the original image
* use processed images, crops, enhancement filters and model outputs to establish its spatial location and extent
* distinguish confident findings from uncertain observations
* do not invent abnormalities
* do not infer clinical history that is not visible in the image

For every focal lesion or abnormality identified, determine its spatial bounding box in the original image coordinates:
* `bbox`: `[x, y, w, h]` where:
  - `x`: top-left corner horizontal pixel coordinate (0 to image_width)
  - `y`: vertical pixel coordinate of the top-left corner (0 to image_height)
  - `w`: width of the bounding box in pixels
  - `h`: height of the bounding box in pixels
* In `scripts/`, you are encouraged to write Python scripts to inspect crops, measure pixel coordinates, threshold areas, or overlay bounding boxes on images in `images/` to verify spatial correctness.
* If multiple abnormalities are visible (for example, bilateral effusion or cardiomegaly with consolidation), produce a distinct localized entry for each.
* If the image is completely normal with no focal abnormalities, `localized_lesions` must be an empty list `[]`.

Compare your final interpretation with `provenance/first_look.md` and record in `provenance/decisions.md` what changed, what did not, and which analysis caused each change.

# 6. FINAL JSON

Write the final result to `<original_filename_without_extension>_analysis/<exact_original_image_filename>.json`, using exactly this structure:

```json
{
  "image": "exact_original_filename.ext",
  "findings": "Relevant visible imaging findings.",
  "impression": "Concise overall impression.",
  "differential_diagnosis": [
    "Primary hypothesis (most likely condition or 'Normal / No acute finding')",
    "Secondary hypothesis",
    "Tertiary hypothesis"
  ],
  "localized_lesions": [
    {
      "label": "Name of pathology (e.g., Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax)",
      "bbox": [x, y, w, h],
      "confidence": "high",
      "reasoning": "Visual description of why this region was bounded here."
    }
  ],
  "limitations": "Important technical or interpretive limitations."
}
```

Rules:

* `"image"` must contain the exact original filename.
* If no definite abnormality is visible, explicitly say so and set `"localized_lesions": []`.
* `"findings"` describes observations, not treatment.
* `"impression"` summarizes the most likely interpretation.
* `"differential_diagnosis"` is an ordered list of diagnostic hypotheses ranked from most likely to least likely, without numerical probabilities.
* `"localized_lesions"` is a list of detected lesions, each containing:
  - `"label"`: the anatomical or pathological name
  - `"bbox"`: a list of exactly four numbers `[x, y, w, h]` in original pixel coordinates (top-left x, top-left y, width w, height h)
  - `"confidence"`: "high", "medium", or "low"
  - `"reasoning"`: concise justification
* `"limitations"` covers image quality, missing views, modality limitations, processing or model limitations, and uncertainty.
* Do not add extra JSON fields beyond the six specified above. Everything else belongs in `provenance/` and `measurements/`.

# 7. FINAL VALIDATION

Before finishing, verify that:

* the final JSON exists, parses, has the correct filename and all required fields
* `"localized_lesions"` contains valid 4-element coordinate lists `[x, y, w, h]` for any identified abnormality
* `scripts/reproduce.py` regenerates every file in `images/` (except `00_original.*`) and in `measurements/` from the original image, as described in the reproducibility section
* you actually ran `scripts/reproduce.py` from a clean state and it succeeded
* every generated image was visually inspected
* `provenance/first_look.md` and `provenance/decisions.md` exist and are complete

Your final response must contain only, for each image: the path of the JSON file, the JSON contents, and a short explanation of how your chosen analyses (including any you discarded) affected the interpretation.
