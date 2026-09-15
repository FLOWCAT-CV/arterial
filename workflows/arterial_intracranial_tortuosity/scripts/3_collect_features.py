#   Copyright 2025 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain

"""
Stage 3: collect per-case centerline features into one table.

Reads the cases table (paths.cases_table_path, .xlsx or .csv) which defines the
cohort — one row per case to collect. Required columns:

    case_id         case directory name (proces_id / pat_id also accepted)
    occlusion_side  left / right (l / r, and Catalan esquerra / dreta, also accepted)

Any additional columns (e.g. occlusion location, M2 dominance) are copied through
to the output unchanged.

For each case the script auto-detects the mode from the case directory contents,
loads the terminal-ICA-to-occlusion centerline on the occlusion side, and writes
its features as one row of paths.features_table_path. For extracranial cases the
cervical ICA centerline features (length_ica, tortuosity_index_ica,
mean_diameter_ica) are appended as well.

Usage:
    python 3_collect_features.py [--cases ID ...] [--skip ID ...]
"""

import os

import pandas as pd

from arterial.io.load_and_save_operations import load_pickle

from pipeline_utils import (
    CASE_ID_COLUMNS,
    MODE_EXTRACRANIAL,
    SIDE_COLUMNS,
    detect_mode_from_outputs,
    find_column,
    load_config,
    normalize_case_id,
    normalize_side,
    parse_batch_args,
    read_table,
    write_table,
)


def centerline_pickle_path(case_dir, mode, centerline_id):
    return os.path.join(case_dir, mode, "individual_centerlines",
                        f"individual_centerline_{centerline_id}.pickle")


if __name__ == "__main__":
    args = parse_batch_args(__doc__)
    config = load_config(args.config)
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

    print(f"Number of cases to collect: {len(cases_df)}")

    rows = []
    missing = []
    for _, case_row in cases_df.iterrows():
        case_id = normalize_case_id(case_row[id_column])
        case_dir = os.path.join(output_dir, case_id)

        side = normalize_side(case_row[side_column])
        if side is None:
            print(f"Skipping {case_id}: unrecognized occlusion side '{case_row[side_column]}'")
            missing.append(case_id)
            continue

        occlusion_centerline_id = f"{side}-tica-occ"
        mode = detect_mode_from_outputs(
            case_dir, [f"individual_centerlines/individual_centerline_{occlusion_centerline_id}.pickle"])
        if mode is None:
            print(f"No arterial analysis found for {case_id} "
                  f"(expected centerline: {occlusion_centerline_id})")
            missing.append(case_id)
            continue

        print(f"Collecting case: {case_id} (mode: {mode}, side: {side})")

        # Start the row with every column of the cases table, then append features.
        row = case_row.to_dict()
        row[id_column] = case_id

        occlusion_centerline = load_pickle(centerline_pickle_path(case_dir, mode, occlusion_centerline_id))
        row.update(occlusion_centerline.graph["features"])

        # Cervical ICA features only exist when a cervico-cerebral CTA was processed.
        ica_pickle = centerline_pickle_path(case_dir, MODE_EXTRACRANIAL, f"{side}-ica")
        if mode == MODE_EXTRACRANIAL and os.path.isfile(ica_pickle):
            ica_features = load_pickle(ica_pickle).graph["features"]
            row["length_ica"] = ica_features["length"]
            row["tortuosity_index_ica"] = ica_features["tortuosity_index"]
            row["mean_diameter_ica"] = ica_features["mean_diameter"]

        rows.append(row)

    features_df = pd.DataFrame(rows)
    write_table(features_df, features_table_path)
    print(f"Wrote {len(features_df)} cases to {features_table_path}")
    if missing:
        print(f"Cases without collected features ({len(missing)}): {missing}")
