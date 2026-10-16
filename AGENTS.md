# AGENTS.md

Guidance for coding agents working in this repository. The user-facing documentation is
[README.md](README.md), with per-stage detail in `docs/modules/`; this file covers what you need to
change the code safely.

## What this is

Arterial is a Python 3.11 package that turns a single head-and-neck CT angiography
(`cta.nii.gz`) into a full vascular analysis: nnU-Net v2 vessel segmentation, VMTK centerlines,
3D U-Net landmark detection, GNN vessel labelling, multi-scale geometric features, catheter pathway
mapping and a GNN (ArterialGNet) that predicts difficult femoral access. Research code from VHIR and
the University of Barcelona, licensed PolyForm Noncommercial 1.0.0.

## Layout

```
arterial/                    the package (this inner directory is what $arterial_dir points at)
├── cli.py                   `arterial` console entry point → run.processor.ArterialProcessor
├── run/processor.py         orchestrates the stages; SUPPORTED_MODES lives here
├── model_registry.py        the ONLY place model-weight paths are resolved
├── segmentation/            VesselSegmenter (nnU-Net v2; TotalSegmentator model splits head/neck)
├── centerline_extraction/   CenterlineExtractor (VMTK); branch extraction runs in a subprocess
├── landmark_detection/      LandmarkDetector (3D U-Net, six bifurcation landmarks)
├── vessel_labelling/        VesselLabeller (PyG GNN on the segments graph)
├── feature_extraction/      FeatureExtractor: local / segment / global features, mapping (supersegments),
│                            vtk_centerline_geometry (per-vessel geometry, carotid bulb)
├── access_prediction/       AccessPredictor (ArterialGNet, 5-fold ensemble, attention maps)
├── io/                      NIfTI/VTK/pickle/JSON load & save; optional DICOM and registration helpers
└── models/                  downloaded weights (gitignored, never commit)
docs/                        installation, per-module docs, Zenodo record text + third-party LICENSE/NOTICE
tests/                       unittest suite (not a package); fixtures in tests/test_data/ (gitignored)
tutorials/scripts/           runnable example entry points
tutorials/notebooks/         executed walkthrough notebooks, one per stage, plus arterial_nb.py helpers
workflows/                   self-contained research pipelines built on the package (scripts + config + README)
download_models.sh           fetches and verifies the weights from Zenodo (record 22694951)
```

Each stage follows the same pattern: a class (`<Stage>er`) in `<stage>/<stage>er.py` with thin public
methods, and the work in sibling `utils.py`, `inference.py`, `preprocessing/`, `postprocessing/`.

`workflows/<name>/` assemble the public classes into end-to-end procedures for a specific study (e.g.
`arterial_intracranial_tortuosity`, which is semi-automatic, with manual review in 3D Slicer). Each has
its own README and is listed in `workflows/README.md`; keep them working when public signatures change.

## Pipeline and data contract

Stages communicate **only through files** in `case_dir/{mode}/`. Each class takes `case_dir` and `mode`,
derives its input and output paths from them, and loads the previous stage's products from disk. That
makes every stage independently runnable (and is what the skip flags and tests rely on). Order, as run
by `ArterialProcessor.perform_analysis`:

1. **Segmentation** → `segmentation.nii.gz`
2. **Centerline extraction** → `segmentations/`, `centerlines/`, `branch_models/`, `branch_model.vtk`,
   `centerline_segments_array.npy`
3. **Landmark detection** → `landmarks.json` (RAS mm), `landmarks_slicer.json`, `individual_centerlines/`
4. **Vessel labelling** → `segments_graph.pickle`, `segments_graph_pred.pickle` (+ `.png`)
5. **Feature extraction** (extracranial only) → `local_graph.pickle`, `single_segments/`, `supersegments/`
6. **Access prediction** (extracranial only) → `access_prediction/femoral_{left,right}/`

Run it with `arterial -cd /path/to/case` (input defaults to `cta.nii.gz` in the case directory). Modes are
`extracranial_vessels` (default, full pipeline) and `intracranial_vessels`; see the README for skip flags.
The full output tree is in the README's "Expected Outputs" section. If you rename or move an output file,
update the README tree, the processor docstrings, the module doc and the tests together; `tests/test_docs.py`
checks the README against a list of key filenames.

Compatibility notes: `--skip_clipping` is accepted but does nothing (the clipped-model code still exists
but is not part of the pipeline). `--cl_dice_nnunet` flips `use_vanilla_nnunet`.

## Environment

- Conda env with Python 3.11 (the README names it `arterial_env`; locally it is `arterial`). Its existence
  does not guarantee that every optional or compiled dependency works. VMTK is **conda-only**
  (`conda install -c conda-forge vmtk`) and cannot go in `setup.py`. PyTorch and the PyG wheels are installed
  by hand first so the right CUDA build is chosen; then `pip install -e .`.
- `arterial_dir` must point at the **inner** package directory (`.../arterial/arterial`), not the repo root.
- Segmentation and landmark detection want an NVIDIA GPU (use a Linux CUDA server). They fall back to CPU
  (very slow; macOS has no CUDA). The other stages are fine on CPU.

## Model weights

- `bash download_models.sh` fetches Zenodo record `22694951` (DOI `10.5281/zenodo.22694951`; no account or
  token). The archive is ~1.1 GB; the script checks the published SHA256 when available, verifies the 18
  checkpoints and required files, and persists `ARTERIAL_MODELS_DIR` in the shell startup file
  (`--no-persist` to avoid that).
- `ARTERIAL_MODELS_DIR` takes precedence over `$arterial_dir/models`.
- The shared layout is already implemented: don't restore per-module checkpoint paths or package weights.

## Invariants that are easy to break

- **Keep stdout clean.** `import arterial` prints its licence banner to **stderr**, and
  `centerline_extraction/postprocessing/run_branch_model_extraction.py` returns a pickled result over
  stdout. Never print to stdout at import time or inside that worker; `tests/test_package.py` checks this.
- **VMTK branch extraction runs in a subprocess** (`branch_and_clipped_model_extraction.py`) because
  VMTK segfaults are uncatchable. Don't move it in-process.
- **Resolve weights through `arterial.model_registry.model_path(...)`**, never by hard-coding paths or
  reading env vars yourself. Keep the resolution order identical to `download_models.sh`.
- **Dependencies:** every third-party import in `arterial/` must be declared in `setup.py`
  (`tests/test_packaging.py` enforces this; add the import→distribution mapping there if the names differ).
  Optional extras (`SimpleITK`, `dicom2nifti`, `antspyx`) must only be imported inside functions in
  `arterial/io/`, never at module level.
- **Every directory with `.py` files needs an `__init__.py`**, or `find_packages` will not install it.
- **Never commit weights or patient data:** `arterial/models/`, `*.pth`, `*.pkl`, `*.model`, `*test_data/`
  and `arterial-zenodo` are gitignored on purpose. Fixtures are anonymised CTAs that are kept local.
- **Third-party licensing:** the `totalsegmentator_mandible` model is Apache-2.0 and its `LICENSE` and `NOTICE`
  (`docs/zenodo/segmentation/totalsegmentator_mandible/`, also under `arterial/segmentation/models/`) must
  travel with every copy of the weights. See `THIRD_PARTY_NOTICES.md` and `docs/zenodo/UPLOADING.md`.

## Code conventions

- Every non-empty source file starts with the two-line header:
  ```python
  #    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
  #    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
  ```
- NumPy-style docstrings (`Parameters` / `Returns` with dashed underlines) on public functions and methods,
  with a third-person summary ("Loads…"), the opening triple quotes on their own line, and a blank line
  before the closing quotes. Some older docstrings misspell `Parmeters`; don't copy that.
- 4-space indentation and long lines are normal here; there is no formatter or linter config. Match the
  surrounding code rather than reformatting files you touch.
- Progress is reported with `print`, not `logging`. Constructors validate `mode` with an `assert`.
- Paths are built with `os.path.join(self.case_dir, self.mode, ...)` in constructors and stored as `*_path`
  attributes.
- Some modules use `from arterial.io.load_and_save_operations import *`. It's fine to keep in existing
  files; prefer explicit imports in new code.
- Coordinates exchanged between stages (landmarks, centerlines) are RAS in millimetres.

## Licensing

Source code is PolyForm Noncommercial 1.0.0, the Arterial weights are CC BY-NC 4.0, and the TotalSegmentator
model is Apache-2.0 (see `LICENSE` and `THIRD_PARTY_NOTICES.md`). Don't change licensing without approval
from the technology-transfer office.

## Testing

The fast tier runs on any machine, but comprehensive testing is only meaningful on a Linux machine with a
CUDA GPU, the model weights and the fixtures. Always report which tier you ran (or that tests were not run),
and don't treat historical results as a current baseline.

The suite uses `unittest` (not pytest). Run it from the repo root with the env active and
`arterial_dir` set:

```bash
ARTERIAL_SKIP_SLOW=1 python -m unittest discover -s tests -v     # fast tier: no weights or GPU needed
python -m unittest discover -s tests -v -p test_feature_geometry.py   # one module
python -u -m unittest discover -s tests -v                        # everything (GPU + weights + fixtures)
```

- `@slow` (from `tests/helpers.py`) marks anything that runs model inference or VMTK. Full-mode centerline
  tests take ~40 min.
- Subclass `helpers.ArterialTestCase`: it gives each test a fresh temp `case_dir`. Use
  `require_fixture(...)` (skips if the fixture is missing) and `stage_fixture(...)` (copies a fixture in as if
  an earlier stage had produced it).
- Prefer synthetic inputs for unit tests of helpers (see `test_*_utils.py`, `test_feature_geometry.py`), so
  they stay in the fast tier.
- `tests/test_products.py` and notebooks 03-07 need derived fixtures: `python tests/make_derived_fixtures.py`.
- `tests/test_docs.py` checks that every ```python block in README.md, `docs/installation.md`, `docs/modules/*.md`
  and `tests/README.md` parses, that its `from arterial... import X` names exist, and that relative links
  resolve. Renaming a public class or function means updating those docs too.
- `tests/test_download_script.py` spins up a local HTTP server, so it needs `bash`, `curl` and `tar`.
- Fixture and slow-test requirements are in `tests/README.md`.

## Docs and tutorials

- The README is the main entry point. Module detail goes in `docs/modules/<module>.md`, which the README
  table links to.
- The notebooks are committed **with executed outputs** so they read as documentation. Re-execute a notebook
  after changing what it demonstrates. They write into `tutorials/notebooks/case/` (gitignored).
- `tutorials/scripts/` must stay runnable against the public classes. Update them when signatures change.

## Git

- Work on a branch (`feat/...`, `docs/...`, `test/...`) and merge to `main` via PR.
- Commit subjects are short, imperative, and often scoped: `README: ...`, `tests: ...`, `notebooks: ...`,
  `download_models.sh: ...`, or `<module>: ...`.
- Stage intended paths explicitly. Keep `dev/`, fixtures, weights and other local data out of commits, and
  preserve local notes and data during cleanup.
