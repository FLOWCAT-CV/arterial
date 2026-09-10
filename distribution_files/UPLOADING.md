# Files that must travel with the weights

The weights themselves are not version-controlled: they live in the Zenodo
archive and are downloaded into `arterial/models/`, which is gitignored.

The *text* files that must accompany them are version-controlled here instead, so the record description and
the third-party notices have a reviewable history:

```
README.md                                      -> attached to the Zenodo record (its description)
segmentation/totalsegmentator_mandible/LICENSE -> Apache-2.0 text for the redistributed model
segmentation/totalsegmentator_mandible/NOTICE  -> attribution and statement of changes
```

The `LICENSE` and `NOTICE` are not optional. Arterial redistributes the `craniofacial_structures`
model from TotalSegmentator under Apache-2.0, and section 4 of that licence requires the licence text
and the attribution notice to be included with **every** copy — whatever the channel. That is why
this directory is named for distribution in general rather than for any one host: the same three
files belong in the Zenodo archive and in any other copy of the weights you hand to someone. See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## Zenodo

`scripts/download_models.sh` reconstructs the weights from an archive built out of
`arterial/models/`, so the `LICENSE` and `NOTICE` are carried along automatically as long as they
were downloaded with the weights in the first place. When building a new archive to deposit, confirm
they are present before uploading:

```bash
tar tzf arterial-models-v1.tar.gz | grep -E 'totalsegmentator_mandible/(LICENSE|NOTICE)'
```

The record description (`README.md`) is not part of the nnU-Net layout, so add it to the Zenodo record as a
separate file alongside the archive.
