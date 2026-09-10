---
license: cc-by-nc-4.0
library_name: arterial
tags:
  - medical
  - medical-imaging
  - cta
  - vascular
  - stroke
  - segmentation
  - nnunet
  - graph-neural-network
extra_gated_heading: Access the Arterial model weights
extra_gated_description: >-
  The Arterial weights are released under CC BY-NC 4.0 for noncommercial research
  use. They are research software, not an approved medical device. One directory,
  segmentation/totalsegmentator_mandible/, is redistributed from TotalSegmentator
  under Apache-2.0 and is not subject to the noncommercial restriction.
extra_gated_prompt: >-
  By requesting access you confirm that you have read and accept the Creative
  Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) licence for
  the Arterial weights, and that you understand these models are provided for
  research purposes only. They have not been cleared or approved by any
  regulatory body and must not be used to inform clinical decisions about
  individual patients. The directory
  segmentation/totalsegmentator_mandible/ is third-party material redistributed
  from TotalSegmentator under the Apache License 2.0; that licence, not CC BY-NC
  4.0, governs your use of it, and the LICENSE and NOTICE files shipped inside
  that directory state its terms.
extra_gated_fields:
  Full name: text
  Affiliation: text
  Country: country
  Intended use: text
  I accept the CC BY-NC 4.0 terms for the Arterial weights and will not use them commercially: checkbox
  I understand these are research models and not an approved medical device: checkbox
extra_gated_button_content: Request access
---

# Arterial — model weights

Trained model weights for [**Arterial**](https://github.com/FLOWCAT-CV/arterial), an AI framework for
automated vascular analysis of CT Angiography (CTA) in the supra-aortic region, developed to support
mechanical thrombectomy planning in acute ischaemic stroke.

This repository contains **only the weights**. The code, installation instructions and documentation
live in the [GitHub repository](https://github.com/FLOWCAT-CV/arterial).

---

## What is in this repository

| Directory | Task | Architecture | Licence |
|---|---|---|---|
| `segmentation/extracranial_vessels/` | Vessel segmentation from the neck to the Circle of Willis | nnU-Net v2 | CC BY-NC 4.0 |
| `segmentation/intracranial_vessels/` | Vessel segmentation within the cranial cavity | nnU-Net v2 | CC BY-NC 4.0 |
| `segmentation/totalsegmentator_mandible/` | Craniofacial structures, used for head/neck splitting | nnU-Net v2 | **Apache-2.0** (third party) |
| `landmark_detection/` | Anatomical landmark detection at vascular bifurcations | 3D U-Net | CC BY-NC 4.0 |
| `vessel_labelling/` | Anatomical naming of centerline segments (e.g. LCCA) | Graph neural network | CC BY-NC 4.0 |
| `access_prediction/` | Catheter accessibility prediction from pathway graphs | ArterialGNet (multi-scale GNN) | CC BY-NC 4.0 |

The segmentation weights account for essentially all of the ~1.2 GB; the graph models are a few MB each.

`segmentation/totalsegmentator_mandible/` is not an Arterial model. It is redistributed unmodified
from [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) under the Apache License 2.0
and carries no noncommercial restriction — see [Third-party models](#third-party-models) below.

---

## Usage

You must accept the licence above before any download will work. Access is granted **automatically**
the moment you accept — there is no waiting period and no manual approval.

### Recommended: the Hugging Face CLI

**1. Install the client**

```bash
pip install huggingface_hub
```

**2. Accept the licence** — click *Agree and access repository* at the top of this page.

**3. Log in**

Create an access token with the **read** role at
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), then:

```bash
hf auth login
```

Paste the token when prompted. This is a one-off step per machine.

**4. Download**

If you have the [Arterial repository](https://github.com/FLOWCAT-CV/arterial) checked out, use the
bundled script — it places the weights where Arterial expects them and verifies the result:

```bash
bash scripts/download_models.sh
```

Otherwise, fetch them directly:

```bash
hf download FLOWCAT-CV/arterial-models --local-dir arterial-models
```

or from Python:

```python
from huggingface_hub import snapshot_download

snapshot_download("FLOWCAT-CV/arterial-models", local_dir="arterial-models")
```

### Telling Arterial where the weights are

By default Arterial looks in `$arterial_dir/models`. To keep the weights anywhere else, set
`ARTERIAL_MODELS_DIR`, which always takes precedence:

```bash
export ARTERIAL_MODELS_DIR="/data/arterial-models"
```

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
access_prediction/     dataset.json, fold_{0..4}/model_weights.pth
landmark_detection/    six_landmarks_2ch.pth, six_landmarks_11_7.pth
segmentation/          extracranial_vessels/, intracranial_vessels/,
                       totalsegmentator_mandible/  (+ LICENSE, NOTICE - Apache-2.0)
vessel_labelling/      extracranial_vessels/
```

The `segmentation/` subdirectories follow the nnU-Net v2 trained-model folder convention
(`plans.json`, `dataset.json` and `fold_*/checkpoint_final.pth`) and are passed directly to
`nnUNetPredictor.initialize_from_trained_model_folder`.

Full installation instructions, including the rest of the Arterial framework, are in the
[repository README](https://github.com/FLOWCAT-CV/arterial#model-weights).

---

## Intended use

These models are intended for **research on vascular morphology and endovascular treatment
planning**, by researchers working with CTA imaging of the supra-aortic region. They are the weights
used by the Arterial framework to produce segmentations, labelled centerlines, geometric features
and accessibility predictions.

### Out of scope

- **Clinical decision-making for individual patients.** These are research models. They have no
  regulatory clearance (CE, FDA or otherwise) and must not be used to guide patient care.
- **Commercial use of any kind** of the Arterial weights, which the CC BY-NC 4.0 licence prohibits.
  Contact the authors if you need commercial terms. This does not extend to
  `segmentation/totalsegmentator_mandible/`, which is third-party Apache-2.0 material.
- **Imaging modalities other than CTA**, and anatomy outside the supra-aortic region.

---

## Limitations

- Trained on CTA acquired in the context of acute ischaemic stroke workup. Performance on other
  populations, indications or acquisition protocols is unknown.
- Sensitive to acquisition parameters, contrast timing and severe artefact. Poor contrast opacification
  degrades segmentation, and downstream centerline, labelling and accessibility outputs inherit those errors.
- The vessel labelling and access prediction models operate on graphs derived from the segmentation
  step, so segmentation failures propagate through the pipeline.
- Not prospectively validated as a decision-support tool.

<!-- TODO: complete from the CMIG paper before announcing the repository —
     training cohort size, centre(s), scanner vendors and acquisition protocols,
     the train/validation/test split, and reported performance per task. -->

---

## License

The Arterial weights are released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
— **noncommercial use only**, with attribution.

Copyright 2022-2026 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

The same terms apply to the Arterial source code. Arterial's dependencies carry their own licences
(nnU-Net and MONAI under Apache-2.0, VMTK under BSD, PyTorch Geometric under MIT), which you must
comply with independently.

### Third-party models

**One directory is not ours and is not noncommercial.**
`segmentation/totalsegmentator_mandible/` is redistributed from
[TotalSegmentator](https://github.com/wasserth/TotalSegmentator) under the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0), copyright the TotalSegmentator
authors (J. Wasserthal et al., University Hospital Basel). Apache-2.0, not CC BY-NC 4.0, governs
what you may do with it — including commercial use. The full licence text and an attribution notice
ship inside that directory as `LICENSE` and `NOTICE`.

| | |
|---|---|
| Task | `craniofacial_structures` (nnU-Net `Dataset115_mandible`) |
| Classes | 1 mandible, 2 teeth_lower, 3 skull, 4 head, 5 sinus_maxillary, 6 sinus_frontal, 7 teeth_upper |
| Configuration | `nnUNetTrainer_DASegOrd0_NoMirroring__nnUNetPlans__3d_fullres`, 0.5 × 0.5 × 0.5 mm |

TotalSegmentator publishes some models openly under Apache-2.0 and gates others behind a licence key
from its authors; `craniofacial_structures` is one of the open ones, which is what allows it to be
redistributed here.

**Changes made in this redistribution.** The weights are unmodified — nothing retrained, fine-tuned or
converted. The directory is renamed to `totalsegmentator_mandible/` so that nnU-Net can load it
directly, only fold 0 is shipped, and TotalSegmentator's own pre- and post-processing is not
included: Arterial calls nnU-Net directly and uses only class 3 (skull), to separate the head from
the neck in a head-and-neck CTA before vessel segmentation.

No endorsement by the TotalSegmentator authors is implied, and the Apache-2.0 licence grants no
rights in the "TotalSegmentator" name.

---

## Citation

If you use these models, please cite:

```bibtex
@article{canals2023vascular,
  title={A fully automatic method for vascular tortuosity feature extraction in the supra-aortic region: Unraveling possibilities in stroke treatment planning},
  author={P. Canals, S. Balocco, O. Díaz, J. Li, A. García-Tornel, A. Tomasello, M. Olivé-Gadea, M. Ribo},
  journal={Computerized Medical Imaging and Graphics},
  volume={104},
  pages={102170},
  year={2023},
  publisher={Elsevier},
  doi={10.1016/j.compmedimag.2022.102170}
}
```

🔗 https://www.sciencedirect.com/science/article/pii/S0895611122001409

### Third-party model citations

If your work uses `segmentation/totalsegmentator_mandible/` — which it does whenever Arterial runs
its default head/neck splitting — please also cite TotalSegmentator, the `craniofacial_structures`
model and nnU-Net, as their authors ask:

```bibtex
@article{wasserthal2023totalsegmentator,
  title={TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images},
  author={Wasserthal, Jakob and Breit, Hanns-Christian and Meyer, Manfred T. and Pradella, Maurice and Hinck, Daniel and Sauter, Alexander W. and Heye, Tobias and Boll, Daniel and Cyriac, Joshy and Yang, Shan and Bach, Michael and Segeroth, Martin},
  journal={Radiology: Artificial Intelligence},
  volume={5},
  number={5},
  year={2023},
  doi={10.1148/ryai.230024}
}

@article{beyer2026craniofacial,
  title={An innovative AI-based dual segmentation application for head surgery},
  author={Beyer, M. and Brasse, A. and Abazi, S. and Beyer, M. and Vinayahalingam, S. and Seifert, L. and Wasserthal, J. and Segeroth, M. and Sharma, N. and Thieringer, F. M.},
  journal={International Journal of Oral and Maxillofacial Surgery},
  year={2026},
  doi={10.1016/j.ijom.2025.11.005}
}

@article{isensee2021nnunet,
  title={nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
  author={Isensee, Fabian and Jaeger, Paul F. and Kohl, Simon A. A. and Petersen, Jens and Maier-Hein, Klaus H.},
  journal={Nature Methods},
  volume={18},
  number={2},
  pages={203--211},
  year={2021},
  doi={10.1038/s41592-020-01008-z}
}
```

---

## Acknowledgments

Developed by the FlowCAT lab, a division of the Stroke Research group at the Vall d'Hebron Research
Institute (VHIR), Barcelona, Spain.

Built on [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), [VMTK](https://github.com/vmtk/vmtk),
[PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) and [MONAI](https://github.com/Project-MONAI/MONAI).

`segmentation/totalsegmentator_mandible/` is the `craniofacial_structures` model from
[TotalSegmentator](https://github.com/wasserth/TotalSegmentator), redistributed unmodified under
Apache-2.0 with thanks to its authors. See [Third-party models](#third-party-models) for the terms
that apply to it.
