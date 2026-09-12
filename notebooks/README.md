# Notebooks

Visual, step-by-step walkthroughs of the pipeline, one per stage, complementing the
production scripts in `../tutorials/`. Each notebook runs top to bottom on the test fixture
case and writes into `notebooks/case/` (gitignored).

- `01_vessel_segmentation` — fast and full nnU-Net modes, the head/neck split and its seam, intracranial mode (GPU, ~6 min).
- `02_centerline_extraction` — islands, surface, endpoints, centerlines, branches, the segments array (~1 min).
- `03_vessel_labelling` — the segments graph and its features, edges to nodes, single model versus ensemble (GPU, seconds).
- `04_feature_extraction` — local graph and hierarchy, local and segment features, arch type, supersegments (~1 min).
- `05_access_prediction` — what ArterialGNet sees, predictions with their uncertainty, attention maps (GPU, seconds).
- `06_landmark_detection` — the model grid, detection, the mirror correction on purpose, refinement, centerlines between landmarks (GPU, ~2 min).
- `07_vessel_geometry` — cross-section radii, curvature and curve ids, the carotid bulb, entering the skull (~1 min).

## Running them

```bash
conda activate arterial_env
export arterial_dir=/path/to/arterial/arterial
python tests/make_derived_fixtures.py     # once: products of the earlier stages, used by notebooks 3-7
jupyter lab notebooks/
```

Inputs come from `tests/test_data/input_test_data/` (see `tests/README.md` for how to obtain it).
Shared helpers (paths, slice viewer, 3D plots) live in `arterial_nb.py`. The committed notebooks carry their
executed outputs, so they read as documentation without running anything.
