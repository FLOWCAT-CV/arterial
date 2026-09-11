# Notebooks

Visual, step-by-step walkthroughs of the processing stages, complementing the production
scripts in `../tutorials/`. Notebooks are numbered in pipeline order; each one starts from
the products of the previous stage (or from the test fixtures) so it can be run on its own.

Planned notebooks are listed in the repository discussion; this directory is the landing
place. Conventions once they exist:

- One notebook per stage, runnable top to bottom on the fixture case in a few minutes.
- Outputs cleared before committing; heavy products written to a temporary case directory.
- Inputs resolved from `tests/test_data/input_test_data/` when available so the notebooks
  double as executable documentation.
