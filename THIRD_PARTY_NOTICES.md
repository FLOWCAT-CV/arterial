# Third-party notices

Arterial's source code is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE); its
own trained weights are released under CC BY-NC 4.0 (see the end of this file). This file lists third-party material that
Arterial **redistributes**, and the third-party software it **depends on**. The two are different in
kind: redistributed material carries its own licence wherever it travels, including inside the
Arterial model weights repository, and that licence is not replaced by Arterial's.

---

## Redistributed material

### TotalSegmentator — `craniofacial_structures` model

| | |
|---|---|
| Where | `<models directory>/segmentation/totalsegmentator_mandible/` (weights), and `arterial/segmentation/models/totalsegmentator_mandible/` in this source repository (upstream nnU-Net metadata and training logs; the checkpoint itself is gitignored) |
| Upstream | [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) |
| Copyright | The TotalSegmentator authors (J. Wasserthal et al., University Hospital Basel) |
| Licence | Apache License 2.0 — [`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt) |
| Task | `craniofacial_structures` (nnU-Net `Dataset115_mandible`) |
| Classes | 1 mandible, 2 teeth_lower, 3 skull, 4 head, 5 sinus_maxillary, 6 sinus_frontal, 7 teeth_upper |

**This model is not covered by Arterial's noncommercial licence.** It reaches you under Apache-2.0,
which permits commercial use, and nothing in Arterial's terms restricts what you may do with that
model on its own. TotalSegmentator publishes some of its models under Apache-2.0 and gates others
behind a licence key obtained from its authors; `craniofacial_structures` belongs to the openly
available Apache-2.0 group, which is what makes this redistribution possible.

**Changes made.** The files are the upstream files, unmodified — nothing retrained, fine-tuned or
converted. The directory is renamed to `totalsegmentator_mandible/` so nnU-Net can load it directly,
only fold 0 is shipped, and TotalSegmentator's own pre- and post-processing is not included —
Arterial calls nnU-Net directly and uses only class 3 (skull), to separate the head from the neck in
a head-and-neck CTA before vessel segmentation. The full statement travels with the model, in the
`NOTICE` and `LICENSE` files inside that directory; keep both with the model in any further copy or
archive, which is what Apache-2.0 §4 requires.

**Citation.** The TotalSegmentator authors ask that users cite the
[TotalSegmentator paper](https://doi.org/10.1148/ryai.230024), the
[craniofacial_structures paper](https://doi.org/10.1016/j.ijom.2025.11.005) and
[nnU-Net](https://doi.org/10.1038/s41592-020-01008-z). Arterial's
[Citation section](README.md#citation) reproduces all three.

No endorsement by the TotalSegmentator authors is implied, and the Apache-2.0 licence grants no
rights in the "TotalSegmentator" name.

---

## Dependencies

These are installed from their own distribution channels rather than redistributed by Arterial, but
you must comply with their licences independently:

| Project | Licence |
|---|---|
| [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) | Apache-2.0 |
| [MONAI](https://github.com/Project-MONAI/MONAI) | Apache-2.0 |
| [VMTK](https://github.com/vmtk/vmtk) | BSD |
| [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) | MIT |

---

## Arterial's own weights

Every other model in the weights repository — `segmentation/extracranial_vessels/`,
`segmentation/intracranial_vessels/`, `landmark_detection/`, `vessel_labelling/` and
`access_prediction/` — was trained by the authors of Arterial and is released under CC BY-NC 4.0:
noncommercial use only, with attribution. Contact the authors for commercial terms.
