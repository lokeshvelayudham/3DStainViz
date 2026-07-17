# StainViz-3D Volumetric Virtual Staining

StainViz-3D extends the upstream PyTorch CycleGAN/pix2pix workflows with research-use volumetric virtual staining for ordered blockface image sequences. The original `pix2pix`, `cycle_gan`, `aligned`, `unaligned`, and `single` modes remain unchanged.

The new workflows are:

- Paired sparse-anchor pix2pix: `--dataset_mode blockface_paired --model stainviz_3d_pix2pix`
- Unpaired ordered-blockface CycleGAN: `--dataset_mode blockface_unaligned --model stainviz_3d_cycle_gan`
- Whole-volume tiled inference with ordered PNG slices, `volume.npy`, `metrics.json`, `provenance.json`, `confidence.npy`, and `manifest_used.csv`

This pipeline is for engineering characterization and research use. It does not make clinical-validity claims and does not define a performance threshold before real specimen-level benchmarking.

## Data Contract

Raw input is assumed to use one specimen/volume per folder. Slice filenames must contain a recoverable numeric index, usually with a regex such as `slice_(?P<index>\d+)`.

Required calibration:

- `z_spacing_um`
- `microns_per_pixel`
- domain label, for example `blockface` or `HE`
- specimen-level split assignment or deterministic seeded split

Manifest records are pseudonymous. The preparation command derives specimen IDs from directory names with a hash and stores relative image paths when the raw root is used as the manifest root. Do not put patient identifiers in folder names, filenames, scanner IDs, or batch IDs.

Prepare a blockface manifest:

```bash
python -m scripts.stainviz.prepare_manifest \
  --raw-root /path/to/raw_blockface \
  --output-manifest manifests/blockface.csv \
  --qc-summary manifests/blockface_qc.json \
  --domain blockface \
  --filename-regex 'slice_(?P<index>\d+)' \
  --z-spacing-um 20 \
  --microns-per-pixel 10 \
  --split-ratios 0.7,0.15,0.15 \
  --seed 13
```

Validate before training or inference:

```bash
python -m scripts.stainviz.validate_manifest \
  --manifest manifests/blockface.csv \
  --manifest-root /path/to/raw_blockface
```

Validation rejects duplicate planes, specimen leakage across splits, non-monotonic z ordering, missing files, incompatible paired targets, inconsistent resolution metadata, and unsupported splits. Planes marked `missing=true` are allowed so inference can preserve missing/corrupt planes as flagged zero-output entries.

## Registration

If slices are already aligned, identity grids must be explicit:

```bash
python -m scripts.stainviz.register_slices identity-grids \
  --manifest manifests/blockface.csv \
  --manifest-root /path/to/raw_blockface \
  --output-dir registration/blockface_identity \
  --output-manifest manifests/blockface_registered.csv \
  --assume-registered
```

For estimated registration, install SimpleITK and use the `estimate` subcommand on fixed/moving pairs. The command writes a normalized `grid_sample(..., align_corners=False)` grid and a post-warp similarity confidence map:

```bash
python -m scripts.stainviz.register_slices estimate \
  --fixed /path/to/fixed_next_slice.tif \
  --moving /path/to/moving_current_slice.tif \
  --output-grid registration/000120_to_000121.npz \
  --output-confidence registration/000120_to_000121_confidence.npy \
  --metric mattes \
  --transform affine \
  --iterations 100
```

Use Mattes mutual information for cross-modality blockface-to-H&E anchor registration and correlation for same-modality adjacent blockface registration. Low-confidence or failed areas should be represented by low or zero confidence maps; the training loss multiplies these maps with tissue masks and neighbor validity.

## Normalization

Fit normalization only from training data and reuse the same JSON for validation/inference:

```bash
python -m scripts.stainviz.fit_normalization \
  --manifest manifests/blockface_registered.csv \
  --manifest-root /path/to/raw_blockface \
  --domain blockface \
  --split train \
  --channels 3 \
  --output checkpoints/stainviz3d_blockface_normalization.json
```

If no normalization JSON is supplied during inference, images are scaled from their image dtype to `[-1, 1]`.

## Paired Sparse-Anchor Training

Train a K=1 paired baseline first:

```bash
bash scripts/stainviz/train_pix2pix_25d.sh \
  --dataroot /path/to/raw_blockface \
  --manifest_path manifests/blockface_registered.csv \
  --manifest_root /path/to/raw_blockface \
  --name stainviz3d_pix2pix_k1 \
  --context_slices 1 \
  --input_nc 3 \
  --output_nc 3 \
  --crop_size 256 \
  --load_size 256 \
  --lambda_L1 100 \
  --lambda_cross_slice 0
```

Then train the 2.5D paired model with sparse registered anchors:

```bash
bash scripts/stainviz/train_pix2pix_25d.sh \
  --dataroot /path/to/raw_blockface \
  --manifest_path manifests/blockface_registered.csv \
  --manifest_root /path/to/raw_blockface \
  --name stainviz3d_pix2pix_k3 \
  --context_slices 3 \
  --input_nc 3 \
  --output_nc 3 \
  --crop_size 256 \
  --load_size 256 \
  --mixed_precision \
  --lambda_L1 100 \
  --lambda_cross_slice 2 \
  --cross_slice_space sobel \
  --init_G_from checkpoints/stainviz3d_pix2pix_k1/latest_net_G.pth
```

The generator accepts `[B,C,H,W]` and `[B,K,C,H,W]`. It uses a shared 2D encoder, sinusoidal physical-z embeddings, masked multi-head attention at the bottleneck, and a 2D decoder. `K=1` is the compatibility path. Training decodes all valid planes, while the discriminator and visualizer use the center prediction.

## Unpaired CycleGAN Training

Unpaired training uses ordered blockface slabs for domain A and unordered H&E images as single-plane domain B unless a genuinely ordered B-domain sequence with valid warps is provided.

```bash
bash scripts/stainviz/train_cyclegan_25d.sh \
  --dataroot /path/to/raw_blockface \
  --manifest_path manifests/blockface_and_he.csv \
  --manifest_root /path/to/data_root \
  --name stainviz3d_cyclegan_k3 \
  --source_domain blockface \
  --target_domain HE \
  --context_slices 3 \
  --input_nc 3 \
  --output_nc 3 \
  --crop_size 256 \
  --load_size 256 \
  --mixed_precision \
  --lambda_cross_A 2 \
  --lambda_cross_B 0 \
  --init_G_A_from checkpoints/stainviz3d_pix2pix_k3/latest_net_G.pth
```

Partial paired-generator loading is rejected unless `--allow_partial_generator_load` is explicit. The loader reports loaded, missing, and unexpected keys.

## Whole-Volume Inference

Inference loads one validated volume, runs overlapping XY tiles and z-slabs, accumulates repeated z predictions, and preserves missing planes as zero-output flagged slices.

```bash
python -m scripts.stainviz.infer_volume \
  --manifest manifests/blockface_registered.csv \
  --manifest-root /path/to/raw_blockface \
  --domain blockface \
  --split test \
  --volume-id spec_abc123_volume \
  --checkpoint checkpoints/stainviz3d_pix2pix_k3/latest_net_G.pth \
  --output-dir outputs/spec_abc123_he \
  --target-stain HE \
  --input-nc 3 \
  --output-nc 3 \
  --context-slices 3 \
  --tile-size 512 \
  --overlap-xy 64 \
  --device cuda \
  --save-tiff
```

Stable outputs:

- `slices/000000.png`, ordered by manifest slice order
- `volume.npy`, float32 `[Z,C,H,W]`
- `metrics.json`, inference metadata including tile grid and z prediction counts
- `provenance.json`, checkpoint hash, code commit, normalization, calibration, exclusions, interpolation, and research-use label
- `confidence.npy`, per-slice prediction count proxy
- `manifest_used.csv`, exact manifest subset

TIFF output is optional. If `tifffile` is unavailable, PNG slices and NumPy output remain the stable fallback.

## Evaluation

Compare K=1 and volumetric checkpoints at specimen level:

```bash
python -m scripts.stainviz.evaluate_volume \
  --baseline-volume outputs/spec_abc123_he_k1/volume.npy \
  --volumetric-volume outputs/spec_abc123_he_k3/volume.npy \
  --target-volume outputs/spec_abc123_anchor_targets/volume.npy \
  --valid-indices 10,42,87 \
  --output reports/spec_abc123_eval.json
```

The report includes SSIM, PSNR, CIEDE2000, intensity correlation on valid paired anchors, adjacent-slice volumetric discontinuity, and a z-smoothing control. Do not interpret smoother z output as automatically better; the smoothing control exists to expose artificially uniform predictions.

## GPU and Patch Defaults

The default target is one 24 GB GPU:

- 256 to 512 px patches
- `K=3` by default, `K=5` for ablation if memory allows
- mixed precision enabled with `--mixed_precision`
- tiled volume inference for full-resolution volumes
- center-plane discriminator supervision

## Out of Scope

This release intentionally excludes uncertainty prediction, learned registration, native 3D convolutional generation, conditional multi-stain training, and a 3D post-generation refiner. The production StainViz web app is not modified; this pipeline provides a stable volume-output contract for later integration.

