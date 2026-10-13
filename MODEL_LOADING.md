# Loading the released MoCaT encoders

## 1. Dependencies and downloads

The release was checked with Python 3.10, PyTorch 2.1.0, NumPy 1.26.x, the SAM2 Hiera implementation, the original ST-MEM implementation, and PTACL's layerscale-enabled transformers fork. Install PyTorch/torchvision appropriate for your platform and use NumPy <2 with this PyTorch version. Other dependencies include scipy, einops, timm, pandas and scikit-learn as required by the upstream implementations.

```bash
git clone https://github.com/frmdshujia/mocat.git
cd mocat
mkdir -p external weights
git clone https://github.com/facebookresearch/sam2.git external/sam2
git clone https://github.com/bakqui/ST-MEM.git external/ST-MEM
git clone https://github.com/Alsalivan/ecgcmr.git external/ecgcmr
pip install -e external/sam2
pip install -e external/ecgcmr/external/transformers
# Install the dependencies required by ST-MEM's upstream instructions.

curl -fL https://github.com/frmdshujia/mocat/releases/download/v1.0.0/mocat_teacher_seed0.pth -o weights/mocat_teacher_seed0.pth
curl -fL https://github.com/frmdshujia/mocat/releases/download/v1.0.0/mocat_stmem_seed0.pth -o weights/mocat_stmem_seed0.pth
curl -fL https://github.com/frmdshujia/mocat/releases/download/v1.0.0/mocat_vitmae_seed0.pth -o weights/mocat_vitmae_seed0.pth
(cd weights && shasum -a 256 -c ../SHA256SUMS)
```

Stock Hugging Face transformers is **not** a substitute for PTACL's fork: the ViT-MAE checkpoint includes layerscale parameters. The loader checks for this and loads all state dictionaries with `strict=True`.

## 2. ECG input and inference

Input is an authorized/user-provided NumPy array of shape `(12,5000)`: twelve leads, ten seconds, 500 Hz. Preserve the original amplitude scale used by your data pipeline. Do not feed a 250-Hz waveform into this interface without adapting the input contract.

**ST-MEM** resamples 500→250 Hz, center-crops 2250 samples, applies high-pass 0.67 Hz and low-pass 40 Hz filters, then standardizes across leads and time using the original ST-MEM transform classes. Output is `(1,768)`.

```bash
python code_available/infer_released.py --model stmem \
  --checkpoint weights/mocat_stmem_seed0.pth \
  --stmem-repo external/ST-MEM \
  --input /path/to/ecg_12x5000.npy --output /path/to/stmem_embedding.npy
```

**ViT-MAE** center-crops samples 1250:3750 from the original 500-Hz input. No resampling, filtering or amplitude normalization is applied in this release path. It averages patch tokens excluding CLS with mask_ratio=0. Output is `(1,512)`.

```bash
python code_available/infer_released.py --model vitmae \
  --checkpoint weights/mocat_vitmae_seed0.pth \
  --input /path/to/ecg_12x5000.npy --output /path/to/vitmae_embedding.npy
```

Both ECG assets contain `ecg`, `pe`, `pc` and `lsc` states. `pe` and `pc` are two-layer Linear→GELU→Linear heads with output dimension 128; ECG input dimensions are 768 for ST-MEM and 512 for ViT-MAE, and the CMR input dimension is 768. For the published frozen representation probes, use the encoder output **before** these projection heads.

## 3. CMR teacher input and inference

Use a preprocessed array of shape `(15,224,224)` or the canonical `(16,224,224)` array (the final T1 frame is ignored). The fifteen frames are ordered:

1. LAX 2Ch: ED, mid, ES;
2. LAX 4Ch: ED, mid, ES;
3. SAX base: ED, mid, ES;
4. SAX mid: ED, mid, ES;
5. SAX apex: ED, mid, ES.

Inputs must already follow the training crop/resize convention and per-frame min-max scaling to `[0,1]`. Do not apply `(x-0.5)/0.5`. Spatial intensity normalization alone does not reproduce the upstream anatomical slice selection/crop process.

Provide an `.npz` with `token_xyz` of shape `(15,16,3)` (or canonical sixteen frames) containing the corresponding 4×4 region-token coordinates in DICOM patient-space millimetres. See `code_available/teacher/geometry.py`; coordinates must correspond to the same frame order, crop and resize as the images.

```bash
python code_available/infer_released.py --model teacher \
  --checkpoint weights/mocat_teacher_seed0.pth \
  --input /path/to/cmr.npy --geometry /path/to/geometry.npz \
  --output /path/to/cmr_embedding.npy
```

Output is the `(1,768)` pooled teacher representation (the second encoder return value). Missing coordinates disable the geometry prior in the architecture; this does not reproduce the full geometry-aware teacher protocol, so the CLI requires geometry.

## 4. Validation and reproducibility boundaries

`LOADING_VERIFIED.json` records strict loading and finite synthetic CPU forward passes for each release asset. Export checks confirmed all encoder tensors remained bit-exact with the original selected checkpoints. This is an inference compatibility check, not a new clinical performance evaluation. The source checkpoints were selected by their existing training validation criteria; no test-driven reselection was performed for release.

Epochs: teacher 21; ST-MEM 3; ViT-MAE 9. All downloads are seed 0. Aggregate results in README refer to existing three-seed scorecards, not a new run or the single asset's guaranteed score.
