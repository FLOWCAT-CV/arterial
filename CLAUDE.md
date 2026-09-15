# Repository guidance

Shared instructions for coding agents. `AGENTS.md` links to this file.

## Project and layout

Arterial analyzes supra-aortic CT angiography for vascular characterization and
endovascular intervention planning. The pipeline combines nnU-Net segmentation,
VMTK centerlines, landmark detection, GNN vessel labelling, feature extraction,
and catheter access prediction.

- `arterial/`: implementation; `arterial/run/processor.py` orchestrates the pipeline.
- `arterial/cli.py`: installed `arterial` command.
- `arterial/model_registry.py`: shared model-path resolution.
- `README.md` and `docs/modules/`: usage and module documentation.
- `tutorials/`: example scripts and notebooks.
- `tests/README.md`: server-side test instructions and fixture requirements.

## Environment and execution

- Target Python 3.11. Follow the platform-specific installation steps in `README.md`;
  VMTK comes from conda-forge, and Arterial is installed with `pip install -e .`.
- The local Conda environment is named `arterial` (`conda activate arterial`).
  Its existence does not guarantee that every optional or compiled dependency works.
- Set `arterial_dir` to the inner package directory: `<repo>/arterial`.
- Run the pipeline with `arterial -cd /path/to/case`; the default input is
  `cta.nii.gz` inside the case directory. See the README for modes and skip flags.
- Use a Linux CUDA server for segmentation and landmark inference.

## Tests

**Do not run tests on this local Mac, including individual or fast tests.**
Testing happens on a separate server. Review code and diffs locally, and report
that tests were not run. Do not treat historical results as a current baseline.

On the test server, use `python -m unittest discover -s tests -v` from the repo
root; `tests/` is not a Python package. `ARTERIAL_SKIP_SLOW=1` selects the fast tier.
See `tests/README.md` for fixtures and slow-test requirements.

## Model weights

- Download with `bash download_models.sh`. The default Zenodo record is `22694951`
  (DOI `10.5281/zenodo.22694951`); no account or token is required.
- The archive is about 1.1 GB. The script checks the published SHA256 when available,
  verifies 18 checkpoints and required files, and persists `ARTERIAL_MODELS_DIR`
  in the shell startup file. Use `--no-persist` to avoid changing shell configuration.
- Resolve all model paths through `arterial.model_registry.model_path(...)`:
  `ARTERIAL_MODELS_DIR` takes precedence over `$arterial_dir/models`.
- The shared layout is already implemented. Do not restore per-module checkpoint
  paths or package model weights. `arterial/models/` is gitignored.
- Keep TotalSegmentator's `LICENSE` and `NOTICE` with its model. The archive's
  version-controlled notice files are in
  `docs/zenodo/segmentation/totalsegmentator_mandible/`.

## Conventions and licensing

- Follow the surrounding code style. Use NumPy-style docstrings with third-person
  summaries ("Loads…"), opening triple quotes on their own line, and a blank line
  before the closing quotes. Describe parameters and return values.
- Preserve the source header; add it to new source files:

  ```python
  #    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
  #    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
  ```

- Source code: PolyForm Noncommercial 1.0.0. Arterial weights: CC BY-NC 4.0.
  The TotalSegmentator model is Apache-2.0. See `LICENSE` and
  `THIRD_PARTY_NOTICES.md`; do not change licensing without technology-transfer
  office approval.
- Keep `dev/`, test fixtures, model weights, and other local data out of commits.
  Preserve local notes and data during cleanup. Stage intended paths explicitly.
