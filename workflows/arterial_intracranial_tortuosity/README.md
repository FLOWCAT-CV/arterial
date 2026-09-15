# Arterial processing pipeline — distal occlusion EVT study

Three sequential scripts that go from a per-case CTA NIfTI to a features table of
CTA-derived vessel-anatomy measurements (tortuosity, length, diameters, ...) of the
access route to a distal intracranial occlusion.

The workflow is **semi-automatic**: the scripts run unattended in batch, but each case
is expected to pass a human quality review. Automatic segmentations and landmarks that
break or misplace a centerline are corrected manually in 3D Slicer and the affected
stage re-run — see [Manual revision](#manual-revision--error-contingency). Plan for
this review pass when budgeting a cohort; the features table is only as reliable as
the per-case QC behind it.

```
1_segmentation_and_landmarks.py   GPU  nnU-Net vessel segmentation + landmark detection
2_centerlines_and_features.py     CPU  centerline extraction + per-centerline features
3_collect_features.py             CPU  consolidate per-case features into one table
4_check_integrity.py              CPU  optional QC: missing files + suspicious feature values
```

All configuration lives in `config.yaml` (edit the paths before running).
Shared helpers live in `pipeline_utils.py`.

![Feature extraction protocol: CTA images and case metadata through automated segmentation & landmark detection, manual correction, centerline extraction, and feature extraction to the features table](figures/Feature%20extraction%20protocol.png)

## Requirements

- Python ≥ 3.9 with the **`arterial`** package installed (provided separately; includes
  the trained nnU-Net segmentation and landmark models) plus `pandas`, `openpyxl`,
  `nibabel`, `pyyaml`, `dicom2nifti`.
- A CUDA GPU for stage 1.

## Data layout

`paths.input_dir` must contain one directory per case, named by the case id:

```
data/
├── 2001/
│   └── cta.nii.gz              # cervico-cerebral CTA (aortic arch to vertex)
├── 2002/
│   └── cta.nii.gz              # head-only CTA also accepted (see mode detection)
└── ...
```

Accepted input filenames: `cta.nii.gz`, `cta.nii`, `cta_intracranial.nii.gz`,
`cta_intracranial.nii`. Input orientation does not matter (stage 1 reorients to LPS).

## Extracranial vs intracranial mode (automatic)

Each case is processed in one of two modes:

- **`extracranial_vessels`** — cervico-cerebral CTA: full route from the cervical ICA
  to the occlusion, including cervical ICA features.
- **`intracranial_vessels`** — head-only CTA: only the terminal-ICA-to-occlusion
  segments can be measured.

With `params.mode: auto` (default) the mode is detected per case, in this order:

1. Existing outputs in the case directory (re-runs keep the original mode).
2. Input filename: `cta_intracranial*` forces intracranial mode.
3. Superior–inferior coverage read from the NIfTI header: ≥ 250 mm
   (`params.extracranial_min_si_coverage_mm`) → extracranial, otherwise intracranial.

The chosen mode and the measured coverage are printed in the batch log — please check
them for the first few cases of a new batch. Set `params.mode` explicitly to override
detection for a whole run.

## Running

```bash
cd scripts
python 1_segmentation_and_landmarks.py
python 2_centerlines_and_features.py
python 3_collect_features.py
python 4_check_integrity.py          # optional QC, see below
```

All scripts accept:

| Flag | Effect |
|---|---|
| `--cases ID [ID ...]` | process only these case ids |
| `--skip ID [ID ...]`  | skip these case ids |
| `--force`             | reprocess even when outputs already exist (stages 1–2) |
| `--config PATH`       | use an alternative config.yaml |

Stages 1 and 2 are resumable: already-completed cases are skipped, and a case that
fails is recorded in `logs_dir/error_log_<stage>_<timestamp>.json` (message + full
traceback) without stopping the batch. Each case also gets its own timestamped log
file inside its case directory.

## Per-case outputs

```
<output_dir>/<case_id>/
├── cta.nii.gz                                  # LPS-reoriented input CTA
├── <mode>/                                     # extracranial_vessels or intracranial_vessels
│   ├── segmentation.nii.gz                     # stage 1: vessel segmentation
│   ├── landmarks.json                          # stage 1: landmarks {label: [x,y,z] RAS mm}
│   ├── landmarks_slicer.json                   # stage 1: same, 3D Slicer markups format
│   └── individual_centerlines/
│       ├── individual_centerline_<id>.vtk      # stage 2: centerline geometry
│       ├── individual_centerline_<id>.pickle   # stage 2: networkx graph w/ features
│       └── individual_centerline_<id>.png      # stage 2: QC render
└── ...
```

Centerline ids (`occ` = occlusion, `tica` = terminal ICA, `eica` = cervical ICA):
`l-ica`, `r-ica`, `l-tica-occ`, `r-tica-occ`, `l-ica-occ`, `r-ica-occ`
(intracranial mode: only `l-tica-occ` / `r-tica-occ`). A centerline is only extracted
when both of its landmarks were detected.

## Manual revision / error contingency

Video walkthrough of the workflow:
[Arterial Workflow: intracranial single vessel tortuosity analysis](https://www.youtube.com/watch?v=JFl8MmSzBCY&list=PLaH-1N45_kaQ).

When a case fails or QC shows a bad result, fix the *inputs* of stage 2 in 3D Slicer
and re-run — never edit stage outputs directly. Stage 2 automatically prefers revised
files through this priority chain (first existing file in `<case_dir>/<mode>/` wins):

| Priority | Segmentation | Landmarks |
|---|---|---|
| 1 (contingency) | `segmentation_mod_minimal.nii.gz` | `landmarks_slicer_mod.json` |
| 2 | `segmentation_mod.nii.gz` | `landmarks.json` (automatic) |
| 3 | `segmentation.nii.gz` (automatic) | |

Typical workflow:

1. **Wrong/missing landmarks** (e.g. occlusion point misplaced): load the CTA and
   `landmarks_slicer.json` in Slicer, move/add points, save as
   `landmarks_slicer_mod.json` in the same `<mode>/` directory.
2. **Segmentation gaps breaking the centerline**: edit `segmentation.nii.gz` in
   Slicer's Segment Editor and save as `segmentation_mod.nii.gz`.
3. **Still failing after full revision** ("incorrigible" cases): start again from the
   automatic `segmentation.nii.gz` and make only the minimal edit needed for the
   centerline to pass; save as `segmentation_mod_minimal.nii.gz`, which outranks
   `segmentation_mod.nii.gz`.

Then re-run only that case:

```bash
python 2_centerlines_and_features.py --cases <case_id> --force
```

The batch log prints which segmentation/landmarks file was used for every case, and a
per-case failure lists the centerline ids that failed in the error log.

## Cases table (stage 3)

`paths.cases_table_path` (.xlsx or .csv) defines the cohort — one row per case to
include in the features table:

| Column | Required | Notes |
|---|---|---|
| `case_id` | yes | case directory name (`proces_id`/`pat_id` also accepted) |
| `occlusion_side` | yes | `left`/`right` (`l`/`r` accepted) |
| anything else | no | copied through to the output unchanged |

Stage 3 loads the `<side>-tica-occ` centerline features for each case, appends
cervical ICA features (`length_ica`, `tortuosity_index_ica`, `mean_diameter_ica`)
when available (extracranial cases), and writes `paths.features_table_path`. Cases
without a collected centerline are listed at the end of the run.

## Integrity check (stage 4, optional)

```bash
python 4_check_integrity.py
```

Read-only QC over the whole cohort — run it after stage 3 (it also works after
stages 1–2, skipping the feature-value checks if the features table doesn't exist
yet). It reports, per case in the cases table:

- **Missing files**: input CTA, segmentation, landmarks, occlusion-side centerline,
  cervical ICA centerline (extracranial cases only), and whether the case reached
  the features table — with a one-line `problem` diagnosis of the first missing
  artifact ("no segmentation (run stage 1)", ...).
- **Suspicious feature values**, two screens:
  - *Plausible ranges*: hard anatomical limits (`PLAUSIBLE_RANGES` at the top of the
    script, e.g. occlusion-path length 15–100 mm, cervical ICA length 110–250 mm).
    Values outside these almost always turned out to be extraction errors in the
    VHIR cohort — a centerline that took a wrong path or a leaked segmentation.
  - *Statistical outliers*: every numeric feature is screened with a robust
    modified z-score (median/MAD, |z| > 3.5), catching cases far from your own
    cohort even where no hard limit exists.

Findings are printed and written next to the features table as
`integrity_files_report` and `integrity_findings_report` (.csv/.xlsx matching the
features table). For each flagged case, inspect the centerline `.png` (or the
segmentation + centerline in Slicer); if the extraction is wrong, fix it via the
manual-revision files above and re-run stage 2 with `--cases <id> --force`, then
stages 3–4 again.
