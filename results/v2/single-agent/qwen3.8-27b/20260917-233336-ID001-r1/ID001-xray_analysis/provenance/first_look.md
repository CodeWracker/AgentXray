# First look (before any processing)

Image: `ID001-xray.png` — 373×454 px, 8-bit RGB (single-channel gray content), min 0 / max 237.

## What the image appears to be

- **Modality:** plain radiograph (X-ray).
- **Body region:** lower leg (distal portion) and foot.
- **Projection:** **lateral** (side) view. Heel/posterior is to the left, toes point to the right and slightly down. The foot is in marked **plantarflexion** (pointed), so the long axis of the foot runs obliquely down-and-forward.
- **Coverage:** partial. The proximal tibia/fibula are cut off at the top edge (only the distal ~1/3 of the tibia and fibula are in view), and the distal phalanges of the toes are cut off at the right edge of the image.

## Structures that deserve attention

1. **Distal tibia and fibula** — only distal parts visible; check for distal tibial/fibular fracture.
2. **Ankle (tibiotalar) joint** — congruence of talar dome under the tibial plafond; check for fracture/dislocation.
3. **Talus** — head, neck, body; check for talar neck/body fracture or dislocation (subtalar/talocrural).
4. **Calcaneus** — posterior and inferior cortex; check for calcaneal fracture, loss of cortical continuity.
5. **Midfoot** (navicular, cuboid, cuneiforms, talonavicular, tarsometatarsal joints) — check for Lisfranc/midfoot fracture-dislocation.
6. **Metatarsals and MTP joints** — check for metatarsal shaft fracture, MTP dislocation. The first MTP shows dense overlapping bone (expected superimposition of metatarsal head + proximal phalanx in lateral view).
7. **Phalanges** — only proximally visible; distal toes out of field.

## What could be abnormal (hypotheses to test)

- No obvious large, gross fracture line or frank dislocation is visible on unaided inspection of the full image and 3–4× zoomed crops (ankle, heel, tarsus, midfoot, forefoot, toes).
- Cortical outlines of the calcaneus, talus, and metatarsals appear continuous at this resolution, but **subtle (hairline) fractures cannot be excluded** at 373×454 px.
- The plantar-flexed positioning itself is a positioning choice, not an abnormality, but it superimposes structures in the forefoot.
- The dense white region at the first MTP could be normal superimposition (most likely) or an overlapped bone fragment (less likely).
- Cannot rule out: subtle metatarsal/phalangeal fracture, tarsal fracture, or small avulsion.

## What limits the image quality

- **Very low resolution** (373×454) for a diagnostic task; fine cortical detail is limited.
- **Single view** (lateral only); no AP/oblique/mediolateral views to cross-check alignment or superimpositions.
- **Truncation**: proximal leg and distal toes out of field.
- Moderate noise/grain; contrast between soft tissue and thin cortical bone is suboptimal in the plantar forefoot.

## Open questions

1. Is there any fracture (distal tibia/fibula, talus, calcaneus, midfoot, metatarsal, phalanx)?
2. Is the ankle and midfoot alignment congruent (no dislocation/subluxation)?
3. Is the dense shadow at the first MTP normal superimposition or a pathological fragment?
4. Can an independent (model-based) read confirm the body region, projection, and absence/presence of fracture?
5. Which pretrained models are even appropriate for a foot radiograph (most in the inventory are chest-trained)?
