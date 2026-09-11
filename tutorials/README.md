# Tutorials

Production-style example scripts. Each one is a complete, runnable entry point built on the
public classes of the package; copy the one closest to your use case and adapt the paths at
the top.

| Script | What it does |
|---|---|
| `arterial_processing_full_pipeline.py` | Runs the whole pipeline on one case from a JSON parameters file (`arterial_processing_params.json`), with per-case logging. |
| `batch_arterial_processing.py` | Template for processing a cohort listed in a spreadsheet, with symlinked inputs and an error log. Set `SRC_DIR`, `DST_DIR` and `XLSX_PATH` first. |
| `landmark_detection_single_case.py` | Landmark detection on one CTA. |
| `landmark_detection_batch_processing.py` | Landmark detection over a directory of cases. |
| `landmark_detection_and_centerline_extraction.py` | Landmarks, then centerlines between landmark pairs, then per-vessel featurisation. |
| `export_vessel_geometry.py` | Exports per-vessel VTK centerlines with cross-section radii, curvature, carotid bulb analysis and optional intracranial transition. |

All scripts expect `arterial_dir` to point at the package directory and the weights to be
installed (see the main README, "Model Weights"). Step-by-step, visual walkthroughs of the
same stages live in `../notebooks/`.
