# Arterial tests

`tests/` is not a package. Run the suite from the repository root with the
`arterial_env` activated and `arterial_dir` pointing at the package directory:

```bash
python -u -m unittest discover -s tests -v            # everything (needs a GPU; VMTK stages are CPU-bound)
ARTERIAL_SKIP_SLOW=1 python -m unittest discover -s tests -v   # fast tier only, under a minute, no models needed
python -m unittest discover -s tests -v -p test_segmenter.py    # one module
```

Every test gets its own temporary case directory (see `helpers.ArterialTestCase`),
so tests are independent of each other and of execution order, and leave nothing
behind.

## Tiers

Tests that run model inference or VMTK carry the `@slow` decorator from
`helpers.py`. They run by default; `ARTERIAL_SKIP_SLOW=1` skips them. Rough
durations on a server with one RTX 3090 and a healthy GPU:

| Module | Fast tier | Slow tier |
|---|---|---|
| `test_segmenter` | seconds | ~1 min fast mode, a few minutes per full-mode test |
| `test_landmark_detector` | seconds | a few minutes per test |
| `test_centerline_extractor` | seconds | ~10 min fast mode, ~40 min full mode (VMTK, CPU) |
| `test_vessel_labeller` | seconds | seconds |
| `test_feature_extractor` | seconds | ~1 min |
| `test_access_predictor` | seconds | seconds |

## Fast, fixture-free modules

`test_io`, `test_packaging`, `test_docs`, `test_processor`, `test_download_script`,
`test_feature_geometry`, `test_global_features`, `test_centerline_utils`,
`test_vtk_geometry`, `test_segmentation_utils`, `test_landmark_utils`,
`test_labelling_utils`, `test_access_utils` and `test_package` run on synthetic inputs
(plus the fixture branch model where noted) and need neither weights nor a GPU.
`test_download_script` starts a local HTTP server and needs `bash`, `curl` and `tar`.

## Fixtures

Inputs live in `tests/test_data/input_test_data/`, which is gitignored (about
240 MB). A test that needs a missing fixture is skipped with the fixture's name
in the skip reason. Obtain the directory from the maintainers or from another
checkout. Contents:

| File | Produced by | Used by |
|---|---|---|
| `cta.nii.gz` | anonymised head-and-neck CTA, 512×512×782 at ~0.5 mm | segmenter, landmark detector, feature extractor, access predictor |
| `segmentation.nii.gz` | `VesselSegmenter` on the CTA, extracranial mode | centerline extractor, landmark detector (2-channel model) |
| `centerline_segments_array.npy` | `CenterlineExtractor.perform_centerline_postprocessing` | vessel labeller, feature extractor |
| `branch_model.vtk` | `CenterlineExtractor.perform_branch_model_extraction` | centerline extractor (loader), feature extractor |
| `segments_graph_pred.pickle` | `VesselLabeller.predict_vessel_types` | vessel labeller (loader), feature extractor |
| `supersegments/*.pickle` | `FeatureExtractor.extract_supersegments` | access predictor |

### Derived fixtures

`tests/test_products.py` needs real execution products. Regenerate them locally (they
are gitignored, ~15 MB) with the weights installed:

```bash
python tests/make_derived_fixtures.py        # writes input_test_data/derived/extracranial_vessels/
```

The model weights must be installed (see the main README, "Model Weights") for
the slow tier; the fast tier only touches the fixtures.
