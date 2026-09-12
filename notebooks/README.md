# Notebooks

Visual, step-by-step walkthroughs of the pipeline, one per stage, complementing the
production scripts in `../tutorials/`. Each notebook runs top to bottom on the test fixture
case and writes into `notebooks/case/` (gitignored).

| Notebook | What it shows | Needs |
|---|---|---|
| `01_getting_started` | Weights resolution, the pipeline driver, anatomy of a finished case | fixtures |
| `02_vessel_segmentation` | Fast and full nnU-Net modes, the head/neck split and seam, intracranial mode | GPU, ~6 min |
| `03_centerline_extraction` | Islands, surface, endpoints, centerlines, branches, the segments array | ~1 min |
| `04_vessel_labelling` | Segments graph, 24 features, node transform, single model vs ensemble | GPU, seconds |
| `05_feature_extraction` | Local graph and hierarchy, local and segment features, arch type, supersegments | ~1 min |
| `06_access_prediction` | What ArterialGNet sees, predictions with uncertainty, attention maps | GPU, seconds |
| `07_landmark_detection` | Model grid, detection, mirror correction on purpose, refinement, centerlines between landmarks | GPU, ~2 min |
| `08_vessel_geometry` | Cross-section radii, curvature and curve ids, carotid bulb, skull entry | ~1 min |
| `09_quality_control` | The checks the pipeline makes and the pictures that expose failures | seconds |
| `10_batch_and_reporting` | Parameters files, the cohort loop, one row per case | seconds |

## Running them

```bash
conda activate arterial_env
export arterial_dir=/path/to/arterial/arterial
python tests/make_derived_fixtures.py     # once: products of the earlier stages, used by notebooks 4-10
jupyter lab notebooks/
```

Inputs come from `tests/test_data/input_test_data/` (see `tests/README.md` for how to obtain it).
Shared helpers (paths, slice viewer, 3D plots) live in `arterial_nb.py`. The committed notebooks carry their
executed outputs, so they read as documentation without running anything.
