# Analysis Decisions

| Question | Method | Why | Potential Changes/Artifacts |
|---|---|---|---|
| Is there a fracture in the bone? | YOLOv8 GRAZPEDWRI-DX | Specifically trained for pediatric wrist trauma (though this is forearm, it might detect bone anomalies). | Might miss mid-shaft forearm fractures if not in training; might flag normal growth plates as anomalies. |
| Is there a fracture? | Bone-Fracture-Detection | General bone fracture classifier. | Might not be optimized for pediatric long bones or forearm specifically. |
| What is the anatomy? | torchxrayvision PSPNet/UNet | Segmentation to confirm anatomical structures. | Might fail if anatomy is highly deformed by fracture. |
| Zero-shot medical description | BiomedCLIP | To see if it identifies "fracture" or "ulna" in context. | Might be too general. |
