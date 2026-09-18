# Decisions

## Analysis 1: Bone Fracture Detection (YOLOv8)
* **Question**: Are there any other fractures in the forearm or wrist that might be subtle?
* **Method**: Use the YOLOv8 GRAZPEDWRI-DX model, which is trained on pediatric wrist trauma.
* **Expected Change**: Detection of a fracture in the radius or other bones would confirm my suspicion.
* **Artifacts/Misleading Effects**: False positives in areas of high density or overlapping bones.
