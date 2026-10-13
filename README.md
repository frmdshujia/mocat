# MoCaT: Motion-Aware Cardiac Teacher

Code and trained encoders for **Breaking the Functional Bottleneck: Motion-Aware Spatio-Temporal CMR Teachers for Cross-Modal ECG Alignment**, accepted at IEEE BIBM 2026.

## Download trained weights

The [v1.0.0 model release](https://github.com/frmdshujia/mocat/releases/tag/v1.0.0) provides three separate inference checkpoints:

- [MoCaT CMR teacher, seed 0](https://github.com/frmdshujia/mocat/releases/download/v1.0.0/mocat_teacher_seed0.pth): geometry-aware Hiera tiny teacher, 15 cine frames, pooled representation of dimension 768; selected epoch 21.
- [MoCaT-aligned ST-MEM ECG encoder, seed 0](https://github.com/frmdshujia/mocat/releases/download/v1.0.0/mocat_stmem_seed0.pth): pooled representation of dimension 768; selected epoch 3.
- [MoCaT-aligned ViT-MAE ECG encoder, seed 0](https://github.com/frmdshujia/mocat/releases/download/v1.0.0/mocat_vitmae_seed0.pth): pooled representation of dimension 512; selected epoch 9.

The ECG checkpoints also contain their ECG/CMR projection heads and learned logit scale. They are full aligned encoder states: downloading original initialization weights is not required for inference, but the upstream architecture implementations are required. Do not interchange ST-MEM and ViT-MAE preprocessing or checkpoints.

See [MODEL_LOADING.md](MODEL_LOADING.md) for dependencies, strict loading, preprocessing, and runnable inference commands. [SHA256SUMS](SHA256SUMS) records download hashes; [model_manifest.json](model_manifest.json) records provenance and selected epochs.

## Which ECG model corresponds to which results?

Existing three-seed scorecards for the **ST-MEM** variant match the main UK Biobank table's displayed phenotype means (functional R² approximately 0.325 and structural R² approximately 0.597). The **ViT-MAE** variant is provided separately; its three-seed means on the same six functional/six structural targets are approximately 0.304 and 0.566. These are distinct model variants, not interchangeable reproductions. The download assets contain representative **seed 0** checkpoints, not a three-seed ensemble; a single seed is not expected to equal an aggregate mean. Only the shared-student comparisons isolate the change of teacher while holding the ECG architecture fixed.

The teacher checkpoint was configured for 25 epochs and selected at epoch 21. Its embedded configuration is authoritative for this release; the original pre-release manuscript description of a 40-epoch budget should not be used to reconstruct this asset.

## Code layout

- `code_available/teacher/`: CMR architecture, geometry, augmentation, training and evaluation.
- `code_available/cross_modal_alignment/`: original ViT-MAE alignment and probing code.
- `code_available/released_models.py`: strict loaders for all three released encoders and exact ECG transforms.
- `code_available/infer_released.py`: inference on user-supplied ECG/CMR arrays; no cohort files required.
- `MODEL_LOADING.md`: release-specific instructions for ST-MEM and ViT-MAE.

The original review scripts remain available for protocol inspection. The released encoder loaders are the verified inference entry points. Full cohort training/evaluation requires authorized data, private subject-level splits, and upstream dependencies; no end-to-end cohort reproduction is claimed without those inputs.

## Data and scope

No UK Biobank images, ECG waveforms, participant identifiers, split tables, labels, embeddings, or individual predictions are included. Obtain UK Biobank and EchoNext data through their respective access processes. This research used the UK Biobank Resource under Application Number **104443**.

These weights support research representation extraction. Downstream regression/classification probes are separate from the encoders. Teacher phenotype-head outputs require the original training target normalization statistics for conversion into physical units; those statistics are not bundled here. This release does not provide a validated clinical diagnostic system.

## Upstream implementations

- [SAM2 / Hiera](https://github.com/facebookresearch/sam2)
- [ST-MEM](https://github.com/bakqui/ST-MEM)
- [PTACL and its custom transformers fork](https://github.com/Alsalivan/ecgcmr)

Respect upstream licenses and usage conditions. No third-party initialization checkpoints are separately redistributed.
