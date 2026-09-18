# ENVIRONMENT (inventory of available tools)

This section describes what is installed. It is an inventory, not a recommendation, and it does not imply that any of these tools is appropriate for the images in this run. Deciding that is part of your task.

## Running Python

Always run Python through the project environment, from any directory:

```
{{PYTHON}} <script.py>
```

Do not install packages and do not use a different interpreter. The environment is pinned and shared by every run of this benchmark, and outputs must be reproducible with it. Computation is CPU-only.

Installed libraries: `numpy`, `scipy`, `opencv-python-headless` (`cv2`), `scikit-image`, `pillow`, `matplotlib`, `pandas`, `torch`, `torchvision` and `torchxrayvision` {{TXV_VERSION}}.

## torchxrayvision

`torchxrayvision` (https://github.com/mlmed/torchxrayvision) is a library of radiograph datasets, preprocessing utilities and pretrained models. Its source code is installed at `{{TXV_SOURCE}}`, and you may read it to understand exactly what each model and function does.

All model weights listed below are already downloaded and verified. There may be no network access, so do not rely on downloading anything else.

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
