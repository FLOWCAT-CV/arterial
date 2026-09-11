#!/usr/bin/env bash
#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#
# Downloads the Arterial model weights from Zenodo.
#
# No account, no token and no licence gate: the weights come down as a single
# archive, verified against the published SHA256, then extracted and checked.
#
# Destination, in order of precedence:
#   1. $ARTERIAL_MODELS_DIR   if set   (any location you like)
#   2. $arterial_dir/models   otherwise (default; gitignored)

set -euo pipefail

# Published Zenodo record of the weights. Override with --record <id> or ZENODO_RECORD.
ZENODO_RECORD="${ZENODO_RECORD:-22694951}"
# Point at https://sandbox.zenodo.org to test against a sandbox record.
ZENODO_SITE="${ZENODO_SITE:-https://zenodo.org}"
ARCHIVE="arterial-models-v1.tar.gz"
PERSIST=1

say()  { printf '%s\n' "$*"; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<USAGE
Usage: bash scripts/download_models.sh [--record ID] [--site URL] [--no-persist] [--help]

Downloads the Arterial model weights from Zenodo.

  --record ID    Zenodo record id (default: ${ZENODO_RECORD}).
  --site URL     Zenodo site (default: ${ZENODO_SITE}).
                 Use https://sandbox.zenodo.org for a sandbox record.
  --no-persist   Do not write ARTERIAL_MODELS_DIR to your shell startup file.
  --help         Show this message.
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --record)     ZENODO_RECORD="${2:?--record needs an id}"; shift ;;
        --site)       ZENODO_SITE="${2:?--site needs a URL}"; shift ;;
        --no-persist) PERSIST=0 ;;
        -h|--help)    usage; exit 0 ;;
        *)            printf 'Unknown option: %s\n\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

case "$ZENODO_RECORD" in
    ""|*[!0-9]*) fail "Invalid Zenodo record id '${ZENODO_RECORD}': pass a numeric id with --record <id>." ;;
esac

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

# -------------------------------------------------------------- sha256 helper
if command -v sha256sum >/dev/null 2>&1; then
    sha256_of() { sha256sum "$1" | awk '{print $1}'; }
elif command -v shasum >/dev/null 2>&1; then
    sha256_of() { shasum -a 256 "$1" | awk '{print $1}'; }   # macOS
else
    sha256_of() { printf ''; }
fi

# ------------------------------------------------------------------ download
BASE="${ZENODO_SITE}/records/${ZENODO_RECORD}/files"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

say ""
say "The Arterial weights are released under CC BY-NC 4.0 - noncommercial use only."
say "  ${ZENODO_SITE}/records/${ZENODO_RECORD}"
say ""
say "One directory is third party and is NOT noncommercial:"
say "  segmentation/totalsegmentator_mandible/  - the craniofacial_structures model"
say "  from TotalSegmentator, redistributed under Apache-2.0. Its LICENSE and"
say "  NOTICE files come with it and must stay with the weights if you copy them."
say ""
say "Downloading ${ARCHIVE} (about 1.1 GB) ..."

# -L follows redirects, -C - resumes a partial file if the server allows it
if ! curl -L -C - --fail --progress-bar \
        -o "${WORK}/${ARCHIVE}" "${BASE}/${ARCHIVE}?download=1"; then
    fail "Download failed. Check that record ${ZENODO_RECORD} exists and is public on ${ZENODO_SITE}."
fi

# --------------------------------------------------------------- verify hash
say ""
# Only a definite 404 means "no checksum published"; any other failure (5xx, proxy,
# network) must not silently downgrade to an unverified install.
http_code="$(curl -sL -o "${WORK}/${ARCHIVE}.sha256" -w '%{http_code}' "${BASE}/${ARCHIVE}.sha256?download=1" || printf '000')"
case "$http_code" in
    200)
        expected="$(awk '{print tolower($1)}' "${WORK}/${ARCHIVE}.sha256")"
        case "$expected" in
            *[!0-9a-f]*|"") fail "Published checksum file is malformed: $(head -c 120 "${WORK}/${ARCHIVE}.sha256")" ;;
        esac
        actual="$(sha256_of "${WORK}/${ARCHIVE}")"
        if [ -z "$actual" ]; then
            say "WARNING: no sha256 tool available; skipping checksum verification." >&2
        elif [ "$expected" = "$actual" ]; then
            say "Checksum OK."
        else
            fail "Checksum mismatch.
       expected ${expected}
       got      ${actual}
       The download is corrupt. Run this script again."
        fi ;;
    404)
        say "No published checksum found (HTTP 404); skipping verification." ;;
    *)
        fail "Could not fetch the published checksum (HTTP ${http_code}). Refusing to install unverified weights; run this script again." ;;
esac

# -------------------------------------------------------------------- extract
say "Extracting ..."
mkdir -p "$DEST"
# macOS archives may carry AppleDouble sidecars; never let them into the models directory.
tar xzf "${WORK}/${ARCHIVE}" -C "$DEST" --exclude='._*' --exclude='.DS_Store'

# --------------------------------------------------------------------- verify
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
    "segmentation/totalsegmentator_mandible/LICENSE" \
    "segmentation/totalsegmentator_mandible/NOTICE" \
    "vessel_labelling/extracranial_vessels/dataset.json" \
    "vessel_labelling/extracranial_vessels/model_weights.pth" ; do
    [ -f "${DEST}/${f}" ] || { say "  missing: $f"; missing=$((missing + 1)); }
done

n_pth=$(find "$DEST" -name "*.pth" -type f ! -name '._*' | wc -l | tr -d ' ')
say "  checkpoints found: ${n_pth} (expected 18)"

if [ "$missing" -ne 0 ] || [ "$n_pth" -ne 18 ]; then
    fail "Verification failed. Delete ${DEST} and run this script again."
fi

# ------------------------------------------------------------------ persist
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

    local n_open n_close
    n_open="$(grep -c '^# >>> arterial models >>>$' "$rc" || true)"
    n_close="$(grep -c '^# <<< arterial models <<<$' "$rc" || true)"
    if [ "$n_open" != "$n_close" ]; then
        say ""
        say "${rc} has an unbalanced arterial models block; not touching it. Add this line by hand:"
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
    # Rewrite in place so the file keeps its mode and stays a symlink if it was one.
    cat "$tmp" > "$rc"
    rm -f "$tmp"

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
