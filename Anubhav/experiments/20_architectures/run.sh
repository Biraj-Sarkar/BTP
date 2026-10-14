#!/bin/bash
# Train three encoder-decoder architectures under identical conditions.
# Same seed, same subject-grouped split, same augmentation, same encoder, same
# epochs — so any difference is the decoder.
set -u
PY=~/btp-local/venv/bin/python
DATA="$HOME/btp-local/datasets/eyes_defy_prepared $HOME/btp-local/datasets/conjunctiva_seg_prepared"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/../.."

for ARCH in unet unetplusplus attention; do
  echo "===== $ARCH ====="
  $PY scripts/train_segmenter.py \
      --data $DATA \
      --arch "$ARCH" \
      --out "$HERE/bundles/$ARCH.pt" \
      --epochs 40 --seed 0 --augment rotation \
      --val-fraction 0.2 --test-fraction 0.2 \
    2>&1 | grep -viE "UserWarning|fetch_version|^  data = |check_version"
done
echo "ALL DONE"
