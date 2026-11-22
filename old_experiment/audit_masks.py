import cv2
import numpy as np
from pathlib import Path
from collections import Counter

root = Path("segmentation_dataset")
mask_dir = root / "masks"

files = sorted(mask_dir.glob("*.png"))

print("Mask count:", len(files))

shapes = Counter()
unique_counts = Counter()
zero_masks = 0
full_masks = 0
empty_masks = 0
ratios = []

for path in files:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print("UNREADABLE:", path.name)
        continue

    shapes[mask.shape] += 1
    unique_counts[len(np.unique(mask))] += 1

    foreground = np.count_nonzero(mask)
    total = mask.size
    ratio = foreground / total
    ratios.append(ratio)

    if foreground == 0:
        empty_masks += 1
    elif foreground == total:
        full_masks += 1

    if ratio == 0:
        zero_masks += 1

print("\nMask dimensions:", shapes)
print("Unique pixel-value counts:", unique_counts)
print("Empty masks:", empty_masks)
print("Full masks:", full_masks)

if ratios:
    print("\nForeground ratio:")
    print("Minimum:", round(min(ratios) * 100, 2), "%")
    print("Maximum:", round(max(ratios) * 100, 2), "%")
    print("Mean:", round(np.mean(ratios) * 100, 2), "%")
    print("Median:", round(np.median(ratios) * 100, 2), "%")
