# RACS2: Deviation Register

Every place the reimplementation cannot match the paper exactly is recorded here.
No entry, no claim.

| Phase | Paper component | What we used | Why | Measured impact | Accepted |
| :-- | :-- | :-- | :-- | :-- | :-- |
| 6 | UniStega calibration data | Official `Unistega.zip` from the DiffStega release linked by paper ref. [46]: `content_prompts` (42 images), `similar_prompts` (30), `style_prompts` (28) | Exact release used by the paper's authors; no gating encountered | Data only, no metric impact | yes |
| 6 | Dataset for reporting | Calibration uses UniStega. All claims use our own evaluation split: COCO val2017 (common, disjoint from the retrieval corpus), `Norod78/Vintage-Faces-FFHQAligned` (facial, FFHQ-derived), Wikimedia Commons cat photographs (style, matching UniStega-Style's AFHQ cat source) | Project decision: paper data is a calibration anchor only, never the evaluation set. `student/FFHQ` is a text-only card and `huggan/AFHQ` exceeds the rows API row-group limit, so aligned alternatives were used | No paper-number delta is computed on evaluation data | yes |
| 6 | LPIPS backbone | `alex` | Paper does not state a backbone | Reported metric depends on backbone; choice is frozen and recorded in every report | yes |
| 6 | UniStega facial subset | 22 face images: 12 FFHQ + 10 CelebA-HQ entries from `similar_prompts/config.yaml` | The released `similar_prompts` folder also contains 8 COCO non-face images (stop sign, motorcycle, juice, building, jockey, donuts, bird, eagle). Including them would make the facial class incoherent | Calibration is approximate, by design | yes |
| 6 | UniStega class counts | 22 per class per role, sampled deterministically | 22 is the number of clean face images in the release; the gate requires >= 20 | Calibration is approximate, by design | yes |
| 6 | Evaluation facial source | Wikimedia Commons `portrait photograph person face` search, filtered by title (artworks, groups, sculptures, synthetic faces) and CLIP (single human face), then canonicalised | Supersedes the `Norod78/Vintage-Faces-FFHQAligned` row above: that dataset contains vintage illustrations, not natural photographs, and CLIP classified 2/6 sampled images as artwork. Commons gives real research-licensed portraits while staying independent of UniStega | Evaluation only; no calibration or paper-number impact | yes |
| 7 | Kpub generation (LLM) | BLIP image captioning (`Salesforce/blip-image-captioning-base`) + fixed template, greedy decoding (`num_beams=1, do_sample=False`) | Paper describes an LLM-generated semantic key; BLIP captioning is deterministic, local, fits 6 GB VRAM, and produces natural-language prompts. No cloud API in the Tx/Rx path | Kpub content may differ from the paper's LLM output; determinism and locality are guaranteed | yes |
| 7 | Common structural feature (OneFormer) | SegFormer-B0 (`nvidia/segformer-b0-finetuned-ade-512-512`) | Paper's OneFormer exceeds the 6 GB VRAM budget on this hardware. SegFormer-B0 produces 150-class ADE20K segmentation maps at 512×512 in fp16 | Different model architecture; same semantic task. Segmentation quality may differ slightly | yes |
| 7 | Facial structural feature (OpenPose) | MediaPipe Face Mesh | `controlnet_aux` OpenPose was unavailable or too heavy for the 6 GB card. MediaPipe provides 468 face landmarks rendered as a skeleton graph | Skeleton topology differs (MediaPipe mesh vs OpenPose body+face). Both provide spatial structure for ControlNet conditioning | yes |

## Rules

- Append only. Never edit or delete a prior row.
- Every row records the phase, the paper component, what was used instead, why, and the
  measured impact.
- Phase 15 must add rows for the reference generator (PicX-Real availability), the ControlNet
  branch (fine-tuned or not), and the semantic codec.
- A calibration result is always labelled approximate. It never enters an evaluation cell or
  a claim table.
