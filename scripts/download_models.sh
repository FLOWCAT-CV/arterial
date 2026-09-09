#!/usr/bin/env bash
#    Copyright 2022-2026 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.
#    SPDX-License-Identifier: CC-BY-NC-4.0
#
# Downloads the Arterial model weights from the Hugging Face Hub.
#
# The weights are gated: you must accept the CC BY-NC 4.0 terms once, on the
# model page, before any download will succeed. This script checks for that and
# tells you what to do if a step is missing.
#
# Destination, in order of precedence:
#   1. $ARTERIAL_MODELS_DIR   if set   (any location you like)
#   2. $arterial_dir/models   otherwise (default; gitignored)

set -euo pipefail

REPO="FLOWCAT-CV/arterial-models"
REPO_URL="https://huggingface.co/${REPO}"
PERSIST=1

usage() {
    cat <<USAGE
Usage: bash scripts/download_models.sh [--no-persist] [--help]

Downloads the Arterial model weights from the Hugging Face Hub.

  --no-persist   Do not write ARTERIAL_MODELS_DIR to your shell startup file.
  --help         Show this message.
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --no-persist) PERSIST=0 ;;
        -h|--help)    usage; exit 0 ;;
        *)            printf 'Unknown option: %s\n\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

say()  { printf '%s\n' "$*"; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- destination
if [ -n "${ARTERIAL_MODELS_DIR:-}" ]; then
    DEST="$ARTERIAL_MODELS_DIR"
    say "Destination : $DEST  (from ARTERIAL_MODELS_DIR)"
elif [ -n "${arterial_dir:-}" ]; then
    DEST="${arterial_dir}/models"
    say "Destination : $DEST  (default, from arterial_dir)"
else
    fail "Neither ARTERIAL_MODELS_DIR nor arterial_dir is set.
       Set arterial_dir to your Arterial package directory, for example:
         export arterial_dir=/path/to/arterial/arterial
       or choose an explicit location:
         export ARTERIAL_MODELS_DIR=/data/arterial-models"
fi

# ------------------------------------------------------------------- hf client
if ! command -v hf >/dev/null 2>&1; then
    fail "The 'hf' command was not found.
       Install the Hugging Face client:
         pip install huggingface_hub"
fi

# ---------------------------------------------------------------------- login
if ! hf auth whoami >/dev/null 2>&1; then
    cat >&2 <<MSG
ERROR: Not logged in to Hugging Face.

  1. Create an access token (role: read) at
       https://huggingface.co/settings/tokens
  2. Log in:
       hf auth login
MSG
    exit 1
fi
say "Logged in   : $(hf auth whoami 2>/dev/null | head -1)"

# --------------------------------------------------------------------- terms
say ""
say "These weights are released under CC BY-NC 4.0 - noncommercial use only."
say "If you have not done so already, open the model page and accept the terms:"
say "  ${REPO_URL}"
say ""

# ------------------------------------------------------------------ download
mkdir -p "$DEST"
say "Downloading ${REPO} (about 1.2 GB) ..."
if ! hf download "$REPO" --local-dir "$DEST"; then
    cat >&2 <<MSG

ERROR: Download failed.

If the error above mentions 403, 401 or 'gated', you have not yet accepted the
licence terms for this model. Open the page below, click 'Agree and access
repository', then run this script again:

  ${REPO_URL}

Access is granted automatically once you accept - there is no waiting period.
MSG
    exit 1
fi

# -------------------------------------------------------------------- verify
say ""
say "Verifying ..."
missing=0
for f in \
    "access_prediction/dataset.json" \
    "access_prediction/fold_0/model_weights.pth" \
    "landmark_detection/six_landmarks_2ch.pth" \
    "landmark_detection/six_landmarks_11_7.pth" \
    "segmentation/extracranial_vessels/nnUNetTrainer__nnUNetPlans__3d_lowres/plans.json" \
    "segmentation/intracranial_vessels/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json" \
    "segmentation/totalsegmentator_mandible/nnUNetTrainer_DASegOrd0_NoMirroring__nnUNetPlans__3d_fullres/plans.json" \
    "vessel_labelling/extracranial_vessels/dataset.json" \
    "vessel_labelling/extracranial_vessels/model_weights.pth" ; do
    [ -f "${DEST}/${f}" ] || { say "  missing: $f"; missing=$((missing + 1)); }
done

n_pth=$(find "$DEST" -name "*.pth" -type f | wc -l | tr -d ' ')
say "  checkpoints found: ${n_pth} (expected 18)"

if [ "$missing" -ne 0 ] || [ "$n_pth" -ne 18 ]; then
    fail "Verification failed. Delete ${DEST} and run this script again."
fi

# ------------------------------------------------------------------ persist
# Writes ARTERIAL_MODELS_DIR into the shell startup file, inside a marked block
# so repeated runs update it in place rather than appending duplicates.
persist_env() {
    local dest="$1" rc shell_name
    shell_name="$(basename "${SHELL:-}")"

    case "$shell_name" in
        zsh)  rc="${ZDOTDIR:-$HOME}/.zshrc" ;;
        bash) if [ "$(uname)" = "Darwin" ] && [ -f "$HOME/.bash_profile" ]; then
                  rc="$HOME/.bash_profile"
              else
                  rc="$HOME/.bashrc"
              fi ;;
        *)    say ""
              say "Could not identify your shell (\$SHELL=${SHELL:-unset})."
              say "Add this line to your shell startup file by hand:"
              say "  export ARTERIAL_MODELS_DIR=\"${dest}\""
              return 0 ;;
    esac

    touch "$rc"
    if [ ! -w "$rc" ]; then
        say ""
        say "${rc} is not writable. Add this line by hand:"
        say "  export ARTERIAL_MODELS_DIR=\"${dest}\""
        return 0
    fi

    local tmp; tmp="$(mktemp)"
    awk '/^# >>> arterial models >>>$/{skip=1} !skip{print} /^# <<< arterial models <<<$/{skip=0}' \
        "$rc" > "$tmp"
    {
        printf '# >>> arterial models >>>\n'
        printf '# Written by scripts/download_models.sh. Safe to edit or remove.\n'
        printf 'export ARTERIAL_MODELS_DIR="%s"\n' "$dest"
        printf '# <<< arterial models <<<\n'
    } >> "$tmp"
    mv "$tmp" "$rc"

    say ""
    say "Added to ${rc}:"
    say "  export ARTERIAL_MODELS_DIR=\"${dest}\""
    say "Run 'source ${rc}' or open a new terminal for it to take effect."
}

say ""
say "Done. Weights installed in ${DEST}"

if [ "$PERSIST" -eq 1 ]; then
    persist_env "$DEST"
else
    say ""
    say "Not persisting (--no-persist). Set this yourself when you need it:"
    say "  export ARTERIAL_MODELS_DIR=\"${DEST}\""
fi
