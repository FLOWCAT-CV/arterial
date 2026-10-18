# Segmentation speed-up

**Default segmentation is 1.6× faster: a median of 48.6 s per case instead of 77.0 s, on 24 head-and-neck CTAs.**
The masks agree with the previous version at a median Dice of 0.998 (worst case 0.994), and the head/neck split is unchanged.
This is now the default; `--resampling scipy` restores the previous resampling.

Measured on one RTX A5000 (24 GB) with nnU-Net 2.8.1 and PyTorch 2.6, over 24 anonymised head-and-neck CTAs:
- 512 × 512 in-plane;
- 0.43–0.59 mm in-plane spacing and 0.4 mm slices;
- 308–385 mm axial coverage.

The default extracranial pipeline was measured: head/neck split, head `3d_fullres`, neck `3d_lowres`.

## What changed

- **Resampling on the GPU.** nnU-Net resamples each image to the model's spacing and back on a single CPU core. It now uses nnU-Net's own PyTorch resampler on the GPU. The model plans are not modified.
- **Models loaded once.** Each nnU-Net model is loaded once per process instead of on every call (three times per case).
- **Faster head/neck split.** The largest-connected-component step on the skull mask uses `cc3d`.

## Time per case

Median seconds per stage. The first case is excluded because it includes loading the models; it took 57 s with the new code and 77 s before.

| Stage | Before | After | Speed-up |
|---|---:|---:|---:|
| Head/neck split | 20.0 | 18.3 | 1.1× |
| Head (`3d_fullres`) | 27.6 | 22.2 | 1.2× |
| Neck (`3d_lowres`) | 26.9 | 6.0 | 4.5× |
| **Total** | **77.0** | **48.6** | **1.6×** |

- **Per case:** the speed-up ranges from 1.4× to 2.2×.
- **Spacing:** the largest gains are on scans whose spacing differs from the head model's (9 of 23). Those need resampling on both sides of the split and go from 104 s to 49 s, which is 2.1×.
- **Mean:** the mean time per case falls from 84.7 s to 49.6 s, which is 1.7×.

## Agreement with the previous version

| Resampling | Dice, median (worst) | Head, worst | Neck, worst | Voxel count difference, median (max) | Split plane |
|---|---|---|---|---|---|
| `torch` (default) | 0.998 (0.994) | 0.988 | 0.994 | 0.10% (0.96%) | identical, 24/24 |
| `scipy` | 1.0000 (0.99999) | 1.0000 | 1.0000 | 2.5 voxels (9) | identical, 24/24 |

- **`torch`:** its interpolation differs slightly from the CPU resampler, so masks change at vessel boundaries. The change is within the acceptance criteria set beforehand: median Dice ≥ 0.99, worst ≥ 0.98.
- **`scipy`:** this keeps the previous resampling. It differs from the previous version by at most 9 voxels per case, out of ~1.4 M vessel voxels. That is the same size as the difference between 1 and 8 CPU threads with identical code.

## Usage

```bash
# Default (GPU resampling)
arterial -cd /path/to/case_dir

# Previous resampling
arterial -cd /path/to/case_dir --resampling scipy

# Several cases, two at a time on one GPU, 4 CPU threads each; rerunning skips completed cases
arterial -cd /path/to/cases/* --workers 2 --num_threads 8
```

- **No GPU:** without a CUDA GPU, `torch` falls back to `scipy`. If the GPU runs out of memory, the prediction is repeated with `scipy`.
- **`--workers N`:**
  - processes N cases at a time;
  - each case needs about 8.5 GB of GPU memory and 16 GB of RAM at its peak, and N is reduced if they don't fit;
  - the throughput gain from several workers was not benchmarked here.
- **`--num_threads`** (or `ARTERIAL_NUM_THREADS`):
  - sets the CPU threads, shared between workers;
  - default: the usable cores, following CPU affinity and container limits, up to 8;
  - it does not change single-case time (73.8 s with 1 thread vs 74.6 s with 8, using `scipy`), but it keeps parallel workers from oversubscribing the CPU.
