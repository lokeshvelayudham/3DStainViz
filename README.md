# StainViz-3D

StainViz-3D is a research-use volumetric virtual staining pipeline for ordered blockface-acquired image sequences. It is designed for datasets produced by sequential blockface imaging systems, where adjacent slices have meaningful acquisition order and spatial adjacency.

The current implementation is a **2.5D volumetric pipeline**, not a native 3D convolutional generator. The model uses a small slab of neighboring planes, usually `K=3` or `K=5`, shares 2D encoder weights across planes, fuses slice context with z-aware attention, and enforces cross-slice consistency during training and inference.

Full workflow documentation is in [docs/stainviz3d.md](docs/stainviz3d.md).

## What This Repository Provides

- Paired sparse-anchor virtual staining from registered H&E/IHC planes.
- Unpaired volumetric CycleGAN training from ordered blockface slabs and unpaired stain-domain images.
- Raw-folder manifest preparation with specimen-safe splits.
- Optional registration preprocessing and confidence maps.
- 2.5D shared-encoder generator with physical-z fusion.
- Cross-slice consistency losses using registration confidence, tissue masks, and source-change gating.
- Deterministic tiled whole-volume inference.
- Stable volume output contract for later StainViz platform integration.
- Specimen-level evaluation metrics and provenance recording.

## 2.5D Versus Native 3D

StainViz-3D uses volumetric context without training a full native 3D generator.

The generator accepts either:

- `K=1`: compatible 2D path
- `K=3`: default 2.5D context
- `K=5`: optional wider z-context when memory allows

Each plane is encoded with shared 2D weights. Bottleneck features are fused with masked multi-head attention using physical z-position embeddings. The decoder produces stain predictions for valid planes, while adversarial supervision uses the center plane for compatibility with the existing 2D discriminator path.

This design targets a single 24 GB GPU with 256 to 512 px patches, mixed precision, and tiled full-volume inference.

## Implemented Workflows

### Paired 2.5D Pix2pix

Use sparse registered H&E/IHC anchors when only some blockface planes have paired targets.

```bash
python train.py \
  --dataset_mode blockface_paired \
  --model stainviz_3d_pix2pix \
  --dataroot /path/to/data_root \
  --manifest_path manifests/blockface_registered.csv \
  --manifest_root /path/to/data_root \
  --context_slices 3 \
  --input_nc 3 \
  --output_nc 3 \
  --crop_size 256 \
  --mixed_precision
```

### Unpaired 2.5D CycleGAN

Use ordered blockface slabs for domain A and unpaired H&E images for domain B. If the target stain domain is unordered, B uses `K=1`.

```bash
python train.py \
  --dataset_mode blockface_unaligned \
  --model stainviz_3d_cycle_gan \
  --dataroot /path/to/data_root \
  --manifest_path manifests/blockface_and_he.csv \
  --manifest_root /path/to/data_root \
  --source_domain blockface \
  --target_domain HE \
  --context_slices 3 \
  --input_nc 3 \
  --output_nc 3 \
  --crop_size 256 \
  --mixed_precision \
  --init_G_A_from checkpoints/stainviz3d_pix2pix_k3/latest_net_G.pth
```

### Whole-Volume Inference

Run deterministic inference over one selected volume. The inference path uses overlapping XY tiles and z-slabs, accumulates repeated z predictions, and preserves missing or corrupt planes as flagged outputs.

```bash
python -m scripts.stainviz.infer_volume \
  --manifest manifests/blockface_registered.csv \
  --manifest-root /path/to/data_root \
  --domain blockface \
  --split test \
  --volume-id spec_abc123_volume \
  --checkpoint checkpoints/stainviz3d_pix2pix_k3/latest_net_G.pth \
  --output-dir outputs/spec_abc123_he \
  --target-stain HE \
  --context-slices 3 \
  --tile-size 512 \
  --overlap-xy 64 \
  --device cuda
```

## Data Preparation

Raw data is expected to use one specimen or volume per directory. Filenames must contain a recoverable numeric plane index.

Prepare a manifest:

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

Validate a manifest before training or inference:

```bash
python -m scripts.stainviz.validate_manifest \
  --manifest manifests/blockface.csv \
  --manifest-root /path/to/raw_blockface
```

Validation rejects duplicate planes, specimen leakage across splits, non-monotonic z ordering, missing files, unavailable paired targets, inconsistent resolution metadata, and incompatible fields. Planes explicitly marked `missing=true` are preserved as missing entries for inference.

## Registration

If slices are already aligned, write explicit identity grids:

```bash
python -m scripts.stainviz.register_slices identity-grids \
  --manifest manifests/blockface.csv \
  --manifest-root /path/to/data_root \
  --output-dir registration/blockface_identity \
  --output-manifest manifests/blockface_registered.csv \
  --assume-registered
```

For estimated registration, install SimpleITK and use the `estimate` subcommand to produce a normalized `grid_sample(..., align_corners=False)` grid and confidence map:

```bash
python -m scripts.stainviz.register_slices estimate \
  --fixed /path/to/fixed_next_slice.tif \
  --moving /path/to/moving_current_slice.tif \
  --output-grid registration/000120_to_000121.npz \
  --output-confidence registration/000120_to_000121_confidence.npy \
  --metric mattes \
  --transform affine
```

## Output Contract

Whole-volume inference writes:

- `slices/000000.png`, ordered by manifest slice order
- `volume.npy`, float32 volume in `[Z,C,H,W]`
- `metrics.json`, tile grid, z prediction counts, and inference metadata
- `provenance.json`, checkpoint hash, code commit, target stain, calibration, exclusions, normalization, and interpolation details
- `confidence.npy`, per-slice prediction count proxy
- `manifest_used.csv`, exact manifest subset used for inference

TIFF output is optional. PNG slices and `volume.npy` are the stable fallback.

## Evaluation

Compare K=1 and volumetric checkpoints with specimen-level metrics:

```bash
python -m scripts.stainviz.evaluate_volume \
  --baseline-volume outputs/spec_abc123_he_k1/volume.npy \
  --volumetric-volume outputs/spec_abc123_he_k3/volume.npy \
  --target-volume outputs/spec_abc123_anchor_targets/volume.npy \
  --valid-indices 10,42,87 \
  --output reports/spec_abc123_eval.json
```

The evaluator reports SSIM, PSNR, CIEDE2000, intensity correlation on paired anchors, adjacent-slice volumetric discontinuity, and a z-smoothing control.

## Repository Layout

- `data/blockface_paired_dataset.py`: paired sparse-anchor slab dataset
- `data/blockface_unaligned_dataset.py`: unpaired blockface/stain slab dataset
- `data/stainviz_manifest.py`: manifest schema, validation, and raw-folder preparation
- `models/stainviz_3d_networks.py`: shared 2.5D generator
- `models/stainviz_3d_pix2pix_model.py`: paired StainViz pix2pix model
- `models/stainviz_3d_cycle_gan_model.py`: unpaired StainViz CycleGAN model
- `models/stainviz_losses.py`: masked reconstruction and cross-slice consistency losses
- `scripts/stainviz/`: preparation, registration, normalization, inference, and evaluation commands
- `util/stainviz_volume_io.py`: tiled whole-volume inference and output packaging
- `docs/stainviz3d.md`: detailed workflow documentation
- `tests/test_stainviz_*.py`: manifest, dataset, generator, loss, model, inference, and option regression tests

## Verification

Run the StainViz test suite:

```bash
python3 -m pytest tests -q
```

Run lint:

```bash
python3 -m flake8 --ignore E501 .
```

## Scope and Limitations

This release intentionally excludes uncertainty prediction, learned registration, native 3D generation, conditional multi-stain training, and a post-generation 3D refiner.

The pipeline is research-use software. It does not make clinical claims, and it does not define fixed performance thresholds before real specimen-level characterization.
