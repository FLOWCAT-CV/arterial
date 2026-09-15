# Workflows

Complete, self-contained pipelines that apply Arterial to specific research tasks. While the
main package provides the building blocks (segmentation, centerline extraction, labelling,
feature extraction), each workflow here assembles them into an end-to-end procedure for a
concrete study — typically a set of scripts, a configuration file, and a README documenting
how to run it and what it produces.

## Available workflows

| Workflow | Description |
| -------- | ----------- |
| [`arterial_intracranial_tortuosity`](arterial_intracranial_tortuosity/) | Semi-automatic: from per-case CTA images to a table of vessel-anatomy features (tortuosity, length, diameters) of the access route to a distal intracranial occlusion, with per-case quality review and manual correction in 3D Slicer. |
