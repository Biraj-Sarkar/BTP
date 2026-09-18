#!/usr/bin/env bash
# Download every dataset this project trains on. See DATASETS.md for what each
# one contains and the licence it carries.
#
# The data is not committed to the repository: ~4.9 GB total, and GitHub rejects
# any single file above 100 MB. This script plus the manifests in
# datasets_manifests/ are the reproducibility contract instead.
#
#   bash scripts/fetch_datasets.sh              # everything
#   bash scripts/fetch_datasets.sh conjunctiva  # just the segmentation source
#   bash scripts/fetch_datasets.sh eyes_defy    # just the Hb source

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASETS="${ROOT}/datasets"
WHICH="${1:-all}"

mkdir -p "${DATASETS}"

# If datasets/ is a real directory inside an iCloud-synced folder, large files
# get evicted and training stalls with `mmap ... errno=60`. A symlink to
# somewhere outside the sync avoids it; see DATASETS.md.
case "$(readlink "${DATASETS}" || echo '')" in
  '') echo "note: datasets/ is a real directory. On macOS with iCloud syncing"
      echo "      Desktop or Documents, point it outside that sync instead:"
      echo "        rmdir datasets && ln -s ~/btp-local/datasets datasets" ;;
esac

fetch_conjunctiva() {
  local dir="${DATASETS}/conjunctiva_seg"
  local zip="${dir}/yxwjgcndg2-1.zip"
  mkdir -p "${dir}"

  if [ -d "${dir}/Images" ]; then
    echo "conjunctiva_seg: already unpacked, skipping"
    return
  fi

  echo "conjunctiva_seg: downloading 4.26 GB from Mendeley (CC BY 4.0)"
  # -C - resumes a partial file rather than starting the 4 GB again.
  curl -L --retry 3 -C - -o "${zip}" \
    "https://data.mendeley.com/public-api/zip/yxwjgcndg2/download/1"
  echo "conjunctiva_seg: unpacking"
  unzip -q -o "${zip}" -d "${dir}"
  echo "conjunctiva_seg: $(find "${dir}" -name '*.jpg' -o -name '*.png' | wc -l) images"
}

fetch_eyes_defy() {
  local dir="${DATASETS}/eyes_defy"

  if [ -d "${dir}" ] && [ -n "$(ls -A "${dir}" 2>/dev/null)" ]; then
    echo "eyes_defy: already present, skipping"
  else
    if ! command -v kaggle >/dev/null 2>&1; then
      echo "eyes_defy: needs the kaggle CLI (pip install kaggle)" >&2
      return 1
    fi
    if [ ! -s "${HOME}/.kaggle/access_token" ] && [ ! -s "${HOME}/.kaggle/kaggle.json" ]; then
      echo "eyes_defy: no Kaggle credentials found." >&2
      echo "  Generate a token at https://www.kaggle.com/settings/api and save" >&2
      echo "  the token string to ~/.kaggle/access_token (chmod 600)." >&2
      return 1
    fi
    echo "eyes_defy: downloading 635 MB from Kaggle"
    kaggle datasets download -d harshwardhanfartale/eyes-defy-anemia \
      -p "${dir}" --unzip
  fi

  # ~45% of this dataset's palpebral annotations are bounding boxes rather than
  # traced contours, so the prepared set is what training should point at.
  if [ ! -d "${DATASETS}/eyes_defy_prepared/images" ]; then
    echo "eyes_defy: filtering annotations and building the prepared set"
    python "${ROOT}/scripts/prepare_eyes_defy.py" \
      --source "${dir}" --out "${DATASETS}/eyes_defy_prepared"
  fi
}

case "${WHICH}" in
  all)          fetch_conjunctiva; fetch_eyes_defy ;;
  conjunctiva)  fetch_conjunctiva ;;
  eyes_defy)    fetch_eyes_defy ;;
  *) echo "usage: $0 [all|conjunctiva|eyes_defy]" >&2; exit 2 ;;
esac

echo
echo "Done. Inspect anything new before training on it:"
echo "  python scripts/inspect_dataset.py datasets/<name>"
