# Installation guide

## System requirements

Python 3.11.

## Operating System
Arterial has been developed and tested on Linux (Ubuntu 22.04), and MacOS (14.X, 15.X).

## Installation

For Ubuntu:

```bash
conda create -n arterial_env python=3.11
conda activate arterial_env
conda install -c conda-forge vmtk

pip install torch==2.6.0 torch_geometric==2.6.1 monai torchio nnunetv2
pip install pyg_lib torch_scatter==2.1.2 torch_sparse==0.6.18 torch_cluster==1.6.3 torch_spline_conv==1.2.2 -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

For MacOS (outdated, not tested):

```bash
conda create -n arterial_env python=3.11
conda activate arterial_env
conda install -c conda-forge vmtk

pip install --upgrade pip
pip install torch==2.2.2 torch_geometric
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
```

To clone repo and install arterial as a Python package:

```bash
git clone https://github.com/FLOWCAT-CV/arterial.git
cd arterial
pip install -e .
```

## Setting up paths

```bash
nano ~/.bashrc
```

```bash
export arterial_dir="/path/to/arterial/arterial"
```

## Model weights

The trained weights are not stored in this repository. They are archived on Zenodo, under a DOI, and
downloaded by a bundled script. No account, no token and no licence gate.

> 📦 **[Zenodo record](https://zenodo.org/records/22694951)** — DOI `10.5281/zenodo.22694951`

### Download

```bash
bash scripts/download_models.sh --record 22694951
```

This downloads about 1.1 GB into `$arterial_dir/models`, verifies the published SHA256, extracts it,
checks that every expected checkpoint and the two Apache-2.0 files arrived, and writes
`ARTERIAL_MODELS_DIR` into your shell startup file:

```bash
# >>> arterial models >>>
export ARTERIAL_MODELS_DIR="/path/to/arterial/arterial/models"
# <<< arterial models <<<
```

The block is marked, so running the script again updates it in place rather than appending a second
copy, and nothing else in the file is touched. Pass `--no-persist` to skip this and set the variable
yourself. Run `source ~/.zshrc`, or open a new terminal, for it to take effect.

Add `--site https://sandbox.zenodo.org` to pull from a sandbox record.

There is no per-file caching, so re-running re-downloads the whole archive; `curl -C -` resumes an
interrupted transfer where the server supports it.

### Installing the weights somewhere else

By default the weights land in `$arterial_dir/models`, which is gitignored. To keep them elsewhere —
a shared drive, a larger volume, a location several checkouts can share — set `ARTERIAL_MODELS_DIR`
before running the script, and keep it set so Arterial can find them afterwards:

```bash
export ARTERIAL_MODELS_DIR="/data/arterial-models"   # add to ~/.bashrc or ~/.zshrc
bash scripts/download_models.sh --record 22694951
```

`ARTERIAL_MODELS_DIR` always takes precedence over the default location.

### Offline and air-gapped machines

Clinical environments frequently have no outbound network access. Download on a connected machine,
copy the directory across, and point `ARTERIAL_MODELS_DIR` at it:

```bash
# on a connected machine
curl -L -o arterial-models-v1.tar.gz \
  "https://zenodo.org/records/22694951/files/arterial-models-v1.tar.gz?download=1"

# on the target machine
mkdir -p /data/arterial-models
tar xzf arterial-models-v1.tar.gz -C /data/arterial-models
export ARTERIAL_MODELS_DIR=/data/arterial-models
```

### Layout

```
<models directory>/
├── access_prediction/     dataset.json, fold_{0..4}/model_weights.pth
├── landmark_detection/    six_landmarks_2ch.pth, six_landmarks_11_7.pth
├── segmentation/          extracranial_vessels/, intracranial_vessels/,
│                          totalsegmentator_mandible/  (+ LICENSE, NOTICE — Apache-2.0)
└── vessel_labelling/      extracranial_vessels/
```

Everything here is an Arterial model under CC BY-NC 4.0 except
`segmentation/totalsegmentator_mandible/`, which comes from
[TotalSegmentator](https://github.com/wasserth/TotalSegmentator) under Apache-2.0. Its `LICENSE` and
`NOTICE` files are downloaded alongside the checkpoint and must stay with it if you copy the weights
to another machine or archive — the tar-and-copy recipe above preserves them. See
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
