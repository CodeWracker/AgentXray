You are analyzing a medical image as a second-opinion support tool for a medical professional.

This is an EXECUTION TASK, not a descriptive task.

You MUST actually use the available filesystem, Python, OpenCV, image-processing libraries, and vision capabilities. Do not merely describe what could be done.

MANDATORY WORKFLOW

1. Locate the input medical image file.

2. Create a dedicated output folder next to the image:

`<original_filename_without_extension>_analysis/`

Example:

Input:
`scan001.png`

Output folder:
`scan001_analysis/`

3. Open and inspect the ORIGINAL image using your own vision capabilities before running image processing.

4. Create and execute a Python script inside the output folder named:

`analyze_image.py`

The script MUST load the original image and generate useful intermediate visualizations.

At minimum, generate:

`01_original.png`
`02_grayscale.png`
`03_contrast_enhanced.png`
`04_sharpened.png`
`05_edge_enhanced.png`

If clinically useful, also generate:

`06_zoom_region_1.png`
`07_zoom_region_2.png`
`08_additional_enhancement.png`

You may use:

* OpenCV
* NumPy
* PIL
* matplotlib
* scipy
* scikit-image
* other relevant computer vision libraries

Do not generate transformations that destroy clinically relevant information.

5. Actually EXECUTE `analyze_image.py`.

6. Verify that the generated image files exist and can be opened.

7. Open and visually inspect EVERY generated intermediate image using your vision capabilities.

Do not infer their contents only from the Python operations.

You MUST visually review:

* the original image
* grayscale image
* contrast-enhanced image
* sharpened image
* edge-enhanced image
* every crop or additional visualization you generated

8. Compare the original image with the processed versions.

Use the processed images only to improve visualization. Do not treat enhancement artifacts as pathological findings.

9. Identify visible medical imaging findings.

For every suspected finding:

* verify that it is visible in the original image
* use processed images only as supporting visualization
* distinguish confident findings from uncertain observations
* do not invent abnormalities
* do not infer clinical history that is not visible in the image

10. Write the FINAL result to a JSON file inside the analysis folder.

The JSON filename MUST be:

`<exact_original_image_filename>.json`

Example:

Input:
`scan001.png`

Output:
`scan001_analysis/scan001.png.json`

11. The JSON MUST be valid JSON and MUST use exactly this simple structure:

```json
{
  "image": "scan001.png",
  "findings": "Relevant visible imaging findings.",
  "impression": "Concise overall impression.",
  "limitations": "Important technical or interpretive limitations."
}
```

Rules:

* `"image"` must contain the exact original filename.
* If no definite abnormality is visible, explicitly say so.
* `"findings"` should describe observations, not treatment.
* `"impression"` should summarize the most likely interpretation and may include a short differential when justified.
* `"limitations"` should mention image quality, missing views, modality limitations, processing limitations, or uncertainty when relevant.
* Do not add extra JSON fields.

12. After writing the JSON file, reopen it from disk and verify that:

* it exists
* it parses as valid JSON
* the filename is correct
* all required fields are present

13. Your final response to the user must contain only:

A. The exact path of the generated JSON file.

B. The JSON contents.

C. A short explanation of the findings and how the processed images affected the interpretation.

Do not provide Python code only without executing it.

Do not stop after creating the script.

Do not stop after generating images.

Do not write the report only in the chat.

The task is complete ONLY when:

* the Python script has been created
* the Python script has been executed
* intermediate images have been generated
* the original and intermediate images have been visually inspected
* the final JSON has been written to disk
* the JSON has been reopened and validated
* the final file path and report have been returned

# HARNESS NOTES (opencode)

You are running inside opencode, without a human in the loop. Nobody will answer questions; make your own decisions and keep going until the task is complete.

Your working directory is `/home/ralph/projects/ufsc/pablo-xray-tests/llm-xray-evaluation/results/v1-runner/single-agent/gemma4-26b/20260917-234221-ID001-r1`. Work only inside it, and do not read or search other directories.

Always run Python with `/home/ralph/projects/ufsc/pablo-xray-tests/llm-xray-evaluation/tools/py <script.py>`. It already has numpy, scipy, OpenCV, scikit-image, Pillow, matplotlib and other scientific libraries installed. Do not install packages and do not use another interpreter.

To look at an image with your own vision, open the image file with the `read` tool. This is the only way to actually see an image; reading its pixel values in Python is not visual inspection.

This complete prompt is saved as `PROMPT.md` in the run directory.
