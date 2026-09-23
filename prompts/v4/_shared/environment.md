# ENVIRONMENT (inventory of available tools)

This section describes what is installed. It is an inventory, not a recommendation, and it does not imply that any of these tools is appropriate for the images in this run. Deciding that is part of your task.

## Running Python

Always run Python through the benchmark interpreter, from any directory:

```
{{PYTHON}} <script.py>
```

It uses a pinned environment shared by every run of this benchmark, with model downloads disabled. Do not install packages and do not use a different interpreter. Computation is CPU-only.

Installed libraries for image processing and numerics: `numpy`, `scipy`, `opencv-python-headless` (`cv2`), `scikit-image`, `pillow`, `matplotlib`, `pandas`, `pywavelets` (`pywt`), `SimpleITK`. Deep learning: `torch`, `torchvision`, `timm`, `transformers`, `open_clip`, `ultralytics`, `torchxrayvision` {{TXV_VERSION}}.

All model weights listed below are already downloaded and verified. There is no network access for models, so do not try to download anything else.

## torchxrayvision

`torchxrayvision` (https://github.com/mlmed/torchxrayvision) is a library of radiograph datasets, preprocessing utilities and pretrained models. Its source code is installed at `{{TXV_SOURCE}}`, and you may read it to understand exactly what each model and function does.

Preprocessing conventions used by all of its models: a single-channel image, with intensities scaled to `[-1024, 1024]` via `xrv.datasets.normalize(img, maxval)`, shaped `(1, H, W)`; `xrv.datasets.XRayCenterCrop()` makes it square and `xrv.datasets.XRayResizer(size)` resizes it. The models expect a batch `(N, 1, size, size)`.

Available pretrained models, all trained on chest radiographs:

| Constructor | What it outputs | Input size |
|---|---|---|
| `xrv.models.DenseNet(weights="densenet121-res224-all")` | 18 pathology scores (`model.pathologies`), trained on NIH, PadChest, CheXpert, MIMIC, Google, OpenI and RSNA | 224 |
| `xrv.models.DenseNet(weights="densenet121-res224-<ds>")`, `<ds>` in `nih`, `pc`, `chex`, `rsna`, `mimic_nb`, `mimic_ch` | pathology scores from a single training dataset; unsupported labels are empty strings | 224 |
| `xrv.models.ResNet(weights="resnet50-res512-all")` | 18 pathology scores, trained on PadChest, NIH, RSNA, SIIM and VinDr | 512 |
| `xrv.autoencoders.ResNetAE(weights="101-elastic")` | `encode`, `decode` and `forward` returning a reconstruction (`out`) and a latent (`z`) | 224 |
| `xrv.baseline_models.chestx_det.PSPNet()` | segmentation of 14 chest structures (`model.targets`) | 512 |
| `xrv.baseline_models.chestx_anatomy.UNetResNet50()` | segmentation of 159 anatomical structures of the chest (`model.targets`) | 512 |
| `xrv.baseline_models.xinario.ViewModel()` | frontal vs lateral chest view logits (`model.targets`) | 224 |
| `xrv.baseline_models.jfhealthcare.DenseNet()` | 5 chest pathology scores (`model.targets`) | 512 |

Classifiers also expose `model.features(x)` for penultimate-layer embeddings. DenseNet and ResNet classifiers apply calibrated per-label operating thresholds by default (see `model.op_threshs` and `xrv.models.op_norm` in the source).

Models that predict demographic attributes (age, sex, race) are deliberately not provided and must not be used.

## Other pretrained models

| Model | Origin and training data | How to load |
|---|---|---|
| BiomedCLIP (vision-language, zero-shot) | Microsoft; ViT-B/16 image encoder and PubMedBERT text encoder trained contrastively on PMC-15M, 15 million figure-caption pairs from PubMed Central articles covering many biomedical image types, radiographs among them. MIT license. | `open_clip.create_model_from_pretrained("local-dir:{{MODELS_DIR}}/hf/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")` returns `(model, preprocess)`; `open_clip.get_tokenizer(<same string>)` returns the tokenizer (context length 256). Compare `model.encode_image` with `model.encode_text` of your own text prompts. |
| MedSAM (promptable segmentation) | Segment Anything ViT-B fine-tuned on about 1.5 million medical image-mask pairs across 10 modalities (Ma et al., Nature Communications 2024). Segments the object inside a bounding box that you provide. Apache-2.0. | `transformers.SamModel.from_pretrained(p)` and `transformers.SamProcessor.from_pretrained(p)` with `p = "{{MODELS_DIR}}/hf/wanglab/medsam-vit-base"`; pass `input_boxes=[[[x0, y0, x1, y1]]]` and `multimask_output=False`. |
| YOLOv8 GRAZPEDWRI-DX (object detection) | Ju & Cai, Scientific Reports 2023; trained on GRAZPEDWRI-DX, about 20 thousand pediatric wrist trauma radiographs from Graz. Classes: `boneanomaly`, `bonelesion`, `foreignbody`, `fracture`, `metal`, `periostealreaction`, `pronatorsign`, `softtissue`, `text`. MIT license. | `ultralytics.YOLO("{{MODELS_DIR}}/files/grazpedwri-yolov8-best.pt").predict(path, imgsz=640, device="cpu")` |
| Bone-Fracture-Detection (image classifier) | Community model published on Hugging Face by an individual contributor (`prithivMLmods/Bone-Fracture-Detection`), fine-tuned from `google/siglip2-base-patch16-224` on the Hugging Face dataset `Hemg/bone-fracture-detection`. No peer-reviewed validation. Labels: `Fractured`, `Not Fractured`. Apache-2.0. | `transformers.AutoModelForImageClassification.from_pretrained(p)` and `transformers.AutoImageProcessor.from_pretrained(p)` with `p = "{{MODELS_DIR}}/hf/prithivMLmods/Bone-Fracture-Detection"` |
