import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import torchxrayvision as xrv
from torchxrayvision.datasets import normalize
import torch.nn.functional as F

base = Path(__file__).resolve().parent.parent
meas = base / "measurements"
meas.mkdir(exist_ok=True)

torch.manual_seed(0)
np.random.seed(0)
torch.use_deterministic_algorithms(True)

img = np.array(Image.open(base / "images" / "00_original.png").convert("L"), dtype=np.float32)

def to_batch(arr, size):
    x = normalize(arr, 255)  # scale to [-1024,1024]
    x = torch.from_numpy(x.astype(np.float32)).float()[None, None, :, :]  # (1,1,H,W)
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    return x  # (1,1,size,size)

results = {}

# E: DenseNet-121 chest pathology (224)
try:
    m = xrv.models.DenseNet(weights="densenet121-res224-all")
    m.eval()
    with torch.no_grad():
        scores = m(to_batch(img, 224))
    labels = m.pathologies
    res = {}
    for l, s in zip(labels, scores.squeeze(0).tolist()):
        if l:
            res[l] = float(s)
    results["densenet121_chest_pathologies"] = res
except Exception as e:
    results["densenet121_error"] = str(e)

# H: ResNetAE 101-elastic autoencoder reconstruction error (224)
try:
    ae = xrv.autoencoders.ResNetAE(weights="101-elastic")
    ae.eval()
    x = to_batch(img, 224)
    with torch.no_grad():
        ret = ae(x)
        out = ret["out"]
        z = ret["z"]
    # reconstruction error map (224x224), downsampled stats
    err = (x - out).abs().squeeze(0).squeeze(0)  # (224,224)
    # crop to the original aspect region of interest is unnecessary; global stats
    err224 = F.interpolate(err.unsqueeze(0).unsqueeze(0), size=(img.shape[0], img.shape[1]), mode="bilinear", align_corners=False).squeeze().numpy()
    err_norm = err224 / (err224.max() + 1e-8)
    results["autoencoder_recon_error"] = {
        "global_mean": float(err.mean()),
        "global_max": float(err.max()),
        "global_p95": float(np.percentile(err, 95)),
        "note": "chest-trained AE on a foot image: high global error expected (OOD), not interpretable as anomaly",
    }
    # save the error map as an image for inspection
    err_img = (np.clip(err_norm, 0, 1) * 255).astype(np.uint8)
    from PIL import Image as PILImage
    PILImage.fromarray(err_img).save(base / "images" / "05_autoencoder_error.png")
except Exception as e:
    results["autoencoder_error_str"] = str(e)

(meas / "05_chest_models_negative_control.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
