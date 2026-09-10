# Hugging Face repository files

The weights themselves live on the Hub, in
[FLOWCAT-CV/arterial-models](https://huggingface.co/FLOWCAT-CV/arterial-models), and are downloaded
into `arterial/models/`, which is gitignored. The *text* files that must travel with them are
version-controlled here instead, so that the model card and the third-party licence notices have a
reviewable history:

```
README.md                                      -> repository root (the model card)
segmentation/totalsegmentator_mandible/LICENSE -> Apache-2.0 text for the redistributed model
segmentation/totalsegmentator_mandible/NOTICE  -> attribution and statement of changes
```

Edit them here, then push:

```bash
hf auth login                     # a token with the write role
hf upload FLOWCAT-CV/arterial-models hf_model_repo . \
    --commit-message "Update model card and third-party notices"
```

`hf upload <repo> <local dir> <path in repo>` uploads the directory contents to the repository root,
preserving the subdirectory layout above. Run it from the repository root.

The same three files must accompany any other distribution of the weights — the Zenodo archive built
by `scripts/download_models_zenodo.sh` included. Apache-2.0 requires the licence text and the
attribution notice to be included with every copy of the TotalSegmentator model, whatever the
channel.
