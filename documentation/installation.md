# Installation guide

## System requirements

## Operating System
Arterial has been developed and tested on Linux (Ubuntu 22.04), and MacOS (14.X, 15.X).

## Installation

For Ubuntu:

```bash
# <=3.9 necessary for vmtk, otherwise it won't install
conda create -n arterial_env python=3.11
conda activate arterial_env
conda install -c conda-forge vmtk

pip install torch==2.6.0 torch_geometric==2.6.1 monai torchio nnunetv2
pip install pyg_lib torch_scatter==2.1.2 torch_sparse==0.6.18 torch_cluster==1.6.3 torch_spline_conv==1.2.2 -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

For MacOS (outdated, not tested):

```bash
# <=3.9 necessary for vmtk, otherwise it won't install
conda create -n arterial_env python=3.9
conda activate arterial_env
conda install -c conda-forge vmtk

pip install --upgrade pip
pip install torch==2.2.2 torch_geometric
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
```

To clone repo and install arterial as a Python package:

```bash
git clone --branch no_slicer https://github.com/FLOWCAT-CV/arterial.git
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

The trained weights are not stored in this repository. They are distributed through the Hugging Face
Hub, gated under CC BY-NC 4.0:

> 🤗 **[FLOWCAT-CV/arterial-models](https://huggingface.co/FLOWCAT-CV/arterial-models)**

Access is granted automatically the moment you accept the terms — there is no waiting period and no
manual approval — but you must accept them once before any download will work.

### Recommended: the Hugging Face CLI

**1. Install the client**

```bash
pip install huggingface_hub
```

**2. Accept the licence**

Open [the model page](https://huggingface.co/FLOWCAT-CV/arterial-models), sign in, and click
*Agree and access repository*. You will be asked to confirm noncommercial use of the Arterial
weights and to acknowledge that these are research models, not an approved medical device. Those
noncommercial terms do not cover `segmentation/totalsegmentator_mandible/`, which is redistributed
from [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) under Apache-2.0.

**3. Log in**

Create an access token with the **read** role at
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), then:

```bash
hf auth login
```

Paste the token when prompted. This is a one-off step per machine.

**4. Download**

```bash
bash scripts/download_models.sh
```

The script downloads about 1.2 GB into `$arterial_dir/models`, then verifies that every expected
checkpoint arrived. If you have not accepted the licence yet, it says so and points you back to
step 2.

It then writes the models location into your shell startup file — `~/.zshrc` for zsh,
`~/.bash_profile` or `~/.bashrc` for bash — so it survives new terminals:

```bash
# >>> arterial models >>>
export ARTERIAL_MODELS_DIR="/path/to/arterial/arterial/models"
# <<< arterial models <<<
```

The block is marked, so running the script again updates it in place rather than appending a second
copy, and nothing else in the file is touched. Pass `--no-persist` to skip this and set the variable
yourself. Run `source ~/.zshrc`, or open a new terminal, for it to take effect.

See the [Hugging Face CLI guide](https://huggingface.co/docs/huggingface_hub/guides/cli) for more
on the client.

### Installing the weights somewhere else

By default the weights land in `$arterial_dir/models`, which is gitignored. To keep them elsewhere —
a shared drive, a larger volume, a location several checkouts can share — set `ARTERIAL_MODELS_DIR`
before running the script, and keep it set so Arterial can find them afterwards:

```bash
export ARTERIAL_MODELS_DIR="/data/arterial-models"   # add to ~/.bashrc or ~/.zshrc
bash scripts/download_models.sh
```

`ARTERIAL_MODELS_DIR` always takes precedence over the default location.

### Offline and air-gapped machines

Clinical environments frequently have no outbound network access. Download on a connected machine,
copy the directory across, and point `ARTERIAL_MODELS_DIR` at it:

```bash
# on a connected machine
hf download FLOWCAT-CV/arterial-models --local-dir arterial-models
tar czf arterial-models.tar.gz arterial-models

# on the target machine
tar xzf arterial-models.tar.gz -C /data
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
