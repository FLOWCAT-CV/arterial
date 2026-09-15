#   Copyright 2025 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain

"""
Stage 4 (optional, read-only): integrity check of the pipeline outputs.

Answers two questions for every case in the cases table:

1. Which files are missing? For each case it reports the presence of the input CTA,
   segmentation, landmarks, the occlusion-side centerline and (extracranial cases)
   the cervical ICA centerline, plus whether the case made it into the features
   table — and summarizes the first missing artifact as the case's `problem`.

2. Which feature values look suspicious? Two complementary screens over the
   features table:
   - PLAUSIBLE_RANGES: anatomically motivated hard limits (units: mm / unitless).
     A value outside its range usually means a segmentation or landmark error
     (e.g. a centerline that took a wrong path), not true anatomy.
   - A robust outlier scan of every numeric feature (modified z-score from the
     median/MAD, |z| > 3.5): flags cases far from the cohort even when no hard
     limit exists for that feature.

Flagged cases should be reviewed visually (centerline .png / Slicer) and, when
wrong, fixed through the manual-revision files and re-run (see README).

Writes two reports next to the features table and prints a summary:
    integrity_files_report.(csv|xlsx)     one row per case: file availability
    integrity_findings_report.(csv|xlsx)  one row per suspicious value

Usage:
    python 4_check_integrity.py [--cases ID ...] [--skip ID ...]
"""

import os

import numpy as np
import pandas as pd

from pipeline_utils import (
    CASE_ID_COLUMNS,
    LANDMARKS_PRIORITY,
    MODE_EXTRACRANIAL,
    SIDE_COLUMNS,
    detect_mode_from_outputs,
    find_column,
    find_input_cta,
    load_config,
    normalize_case_id,
    normalize_side,
    parse_batch_args,
    read_table,
    write_table,
)

# Hard plausibility limits, as {feature: (min, max)} with None = unbounded.
# Derived from cohort review at VHIR; values outside these almost always turned
# out to be extraction errors rather than extreme anatomy.
PLAUSIBLE_RANGES = {
    "length": (15, 100),                      # terminal-ICA-to-occlusion path (mm)
    "length_ica": (110, 250),                 # cervical ICA (mm)
    "tortuosity_index_ica": (0.1, None),      # near-straight cervical ICA = truncated centerline
    "mean_diameter": (None, 10),              # mm; larger = leaked segmentation
    "mean_diameter_ica": (None, 6),           # mm
    "tortuosity_index_last_10mm": (None, 0.4),
    "curvature_energy": (None, 3.7),
}

OUTLIER_MODIFIED_Z = 3.5
MIN_VALUES_FOR_OUTLIER_SCAN = 8


def check_case_files(case_id, side, input_dir, output_dir):
    """File availability for one case. Missing side disables the side-specific checks."""
    case_dir = os.path.join(output_dir, case_id)
    mode = detect_mode_from_outputs(case_dir, LANDMARKS_PRIORITY + ["segmentation.nii.gz"])
    mode_dir = os.path.join(case_dir, mode) if mode else None

    def in_mode_dir(name):
        return mode_dir is not None and os.path.isfile(os.path.join(mode_dir, name))

    def has_centerline(centerline_id):
        return in_mode_dir(os.path.join("individual_centerlines",
                                        f"individual_centerline_{centerline_id}.pickle"))

    row = {
        "case_id": case_id,
        "side": side,
        "mode": mode,
        "has_input_cta": find_input_cta(os.path.join(input_dir, case_id)) is not None,
        "has_segmentation": in_mode_dir("segmentation.nii.gz"),
        "has_segmentation_mod": in_mode_dir("segmentation_mod.nii.gz"),
        "has_segmentation_mod_minimal": in_mode_dir("segmentation_mod_minimal.nii.gz"),
        "has_landmarks_auto": in_mode_dir("landmarks.json"),
        "has_landmarks_mod": in_mode_dir("landmarks_slicer_mod.json"),
        "has_occlusion_centerline": side is not None and has_centerline(f"{side}-tica-occ"),
        # Cervical ICA centerlines only exist for cervico-cerebral (extracranial) scans:
        # None (not False) for intracranial cases so they don't count as missing.
        "has_ica_centerline": (has_centerline(f"{side}-ica")
                               if mode == MODE_EXTRACRANIAL and side is not None else None),
    }

    # First missing artifact in pipeline order, as a one-word diagnosis.
    if side is None:
        row["problem"] = "bad side in cases table"
    elif not row["has_input_cta"] and mode is None:
        row["problem"] = "no input CTA (check input_dir and filename)"
    elif not row["has_segmentation"]:
        row["problem"] = "no segmentation (run stage 1)"
    elif not (row["has_landmarks_auto"] or row["has_landmarks_mod"]):
        row["problem"] = "no landmarks (run stage 1)"
    elif not row["has_occlusion_centerline"]:
        row["problem"] = "no occlusion centerline (run stage 2 / check landmarks)"
    elif mode == MODE_EXTRACRANIAL and not row["has_ica_centerline"]:
        row["problem"] = "no cervical ICA centerline (check eica/tica landmarks)"
    else:
        row["problem"] = ""
    return row


def range_findings(features_df, id_column):
    findings = []
    for column, (low, high) in PLAUSIBLE_RANGES.items():
        if column not in features_df.columns:
            continue
        values = pd.to_numeric(features_df[column], errors="coerce")
        bad = pd.Series(False, index=features_df.index)
        if low is not None:
            bad |= values < low
        if high is not None:
            bad |= values > high
        for _, row in features_df[bad].iterrows():
            findings.append({
                "case_id": normalize_case_id(row[id_column]),
                "check": "plausible_range",
                "column": column,
                "value": row[column],
                "expected": f"[{low if low is not None else '-inf'}, "
                            f"{high if high is not None else 'inf'}]",
            })
    return findings


def outlier_findings(features_df, id_column, feature_columns, already_flagged):
    findings = []
    for column in feature_columns:
        values = pd.to_numeric(features_df[column], errors="coerce")
        valid = values.dropna()
        if len(valid) < MIN_VALUES_FOR_OUTLIER_SCAN or valid.nunique() <= 2:
            continue
        median = valid.median()
        mad = (valid - median).abs().median()
        if mad == 0:
            continue
        modified_z = 0.6745 * (values - median) / mad
        for idx in features_df.index[modified_z.abs() > OUTLIER_MODIFIED_Z]:
            case_id = normalize_case_id(features_df.loc[idx, id_column])
            if (case_id, column) in already_flagged:
                continue
            findings.append({
                "case_id": case_id,
                "check": "outlier",
                "column": column,
                "value": features_df.loc[idx, column],
                "expected": f"cohort median {median:.3g} (modified z = {modified_z[idx]:.1f})",
            })
    return findings


# Artifacts every case must have vs optional manual-revision files (informational).
REQUIRED_ARTIFACTS = ["has_input_cta", "has_segmentation", "has_landmarks",
                      "has_occlusion_centerline", "has_ica_centerline"]
OPTIONAL_ARTIFACTS = ["has_landmarks_auto", "has_landmarks_mod",
                      "has_segmentation_mod", "has_segmentation_mod_minimal"]


def print_availability_summary(files_df):
    print(f"\n=== File availability ({len(files_df)} cases) ===")
    # Either landmarks file satisfies the requirement (revised cases may only keep _mod).
    files_df = files_df.assign(has_landmarks=files_df["has_landmarks_auto"]
                               | files_df["has_landmarks_mod"])
    print("  Required artifacts:")
    for column in REQUIRED_ARTIFACTS:
        applicable = files_df[files_df[column].notna()]  # ICA only applies to extracranial cases
        missing = applicable.loc[~applicable[column].astype(bool), "case_id"].tolist()
        label = column.removeprefix("has_")
        print(f"    {label:28s} {len(applicable) - len(missing):4d}/{len(applicable)}"
              + (f"   missing: {missing}" if missing else ""))
    print("  File variants present (informational):")
    for column in OPTIONAL_ARTIFACTS:
        print(f"    {column.removeprefix('has_'):28s} {int(files_df[column].sum()):4d}/{len(files_df)}")

    problems = files_df[files_df["problem"] != ""]
    if len(problems):
        print(f"\n  Cases with problems ({len(problems)}):")
        for _, row in problems.iterrows():
            print(f"    {row['case_id']}: {row['problem']}")
    else:
        print("\n  All cases have all expected files.")


def print_findings_summary(findings_df):
    print(f"\n=== Suspicious feature values ({len(findings_df)} findings) ===")
    if not len(findings_df):
        print("  None.")
        return
    for check in ("plausible_range", "outlier"):
        subset = findings_df[findings_df["check"] == check]
        if not len(subset):
            continue
        title = ("Outside plausible range (likely extraction errors)"
                 if check == "plausible_range" else
                 f"Statistical outliers (|modified z| > {OUTLIER_MODIFIED_Z}, review advised)")
        print(f"\n  {title}:")
        for _, row in subset.iterrows():
            print(f"    {row['case_id']}  {row['column']} = {row['value']:.4g}   expected {row['expected']}")


if __name__ == "__main__":
    args = parse_batch_args(__doc__)
    config = load_config(args.config)
    input_dir = config["paths"]["input_dir"]
    output_dir = config["paths"]["output_dir"]
    cases_table_path = config["paths"]["cases_table_path"]
    features_table_path = config["paths"]["features_table_path"]

    cases_df = read_table(cases_table_path)
    id_column = find_column(cases_df, CASE_ID_COLUMNS, "case id")
    side_column = find_column(cases_df, SIDE_COLUMNS, "occlusion side")
    if args.cases:
        cases_df = cases_df[cases_df[id_column].astype(str).isin(args.cases)]
    if args.skip:
        cases_df = cases_df[~cases_df[id_column].astype(str).isin(args.skip)]
    print(f"Checking {len(cases_df)} cases from {cases_table_path}")

    # --- 1. File availability ---
    file_rows = []
    for _, case_row in cases_df.iterrows():
        case_id = normalize_case_id(case_row[id_column])
        side = normalize_side(case_row[side_column])
        file_rows.append(check_case_files(case_id, side, input_dir, output_dir))
    files_df = pd.DataFrame(file_rows)

    # --- 2. Feature values ---
    findings = []
    if os.path.isfile(features_table_path):
        features_df = read_table(features_table_path)
        features_id_column = find_column(features_df, CASE_ID_COLUMNS, "case id")
        features_df = features_df[features_df[features_id_column].map(normalize_case_id)
                                  .isin(files_df["case_id"])]

        in_features = set(features_df[features_id_column].map(normalize_case_id))
        files_df["in_features_table"] = files_df["case_id"].isin(in_features)
        not_collected = files_df.loc[~files_df["in_features_table"], "case_id"].tolist()

        findings = range_findings(features_df, features_id_column)
        already_flagged = {(f["case_id"], f["column"]) for f in findings}
        # Only scan pipeline-generated numeric columns, not cases-table pass-through ones.
        feature_columns = [c for c in features_df.select_dtypes(include=[np.number]).columns
                           if c not in cases_df.columns]
        findings += outlier_findings(features_df, features_id_column, feature_columns, already_flagged)
    else:
        not_collected = None
        print(f"\nFeatures table not found ({features_table_path}) — "
              "run 3_collect_features.py first; skipping feature-value checks.")

    findings_df = pd.DataFrame(findings, columns=["case_id", "check", "column", "value", "expected"])

    # --- Report ---
    print_availability_summary(files_df)
    if not_collected is not None:
        print(f"\n=== Features table coverage ===")
        print(f"  Cases missing from the features table: {not_collected if not_collected else 'none'}")
    print_findings_summary(findings_df)

    extension = ".csv" if features_table_path.endswith(".csv") else ".xlsx"
    report_dir = os.path.dirname(features_table_path)
    files_report_path = os.path.join(report_dir, f"integrity_files_report{extension}")
    findings_report_path = os.path.join(report_dir, f"integrity_findings_report{extension}")
    write_table(files_df, files_report_path)
    write_table(findings_df, findings_report_path)
    print(f"\nReports written to:\n  {files_report_path}\n  {findings_report_path}")
