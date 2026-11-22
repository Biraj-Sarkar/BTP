"""Deterministic image/mask discovery, validation, and grouped splits."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import cv2
import numpy as np
SUFFIXES={".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
@dataclass(frozen=True)
class Pair:
    image: Path; mask: Path; group: str
def _group(path):
    parent=path.parent.name.lower()
    return parent if parent not in {"images","image","masks","mask","labels","label","segmented"} else path.stem.rsplit("_",1)[0]
def discover_pairs(root):
    root=Path(root)
    if not root.is_dir(): raise FileNotFoundError(root)
    image_dirs=[d for d in root.rglob("*") if d.is_dir() and "image" in d.name.lower()]
    mask_dirs=[d for d in root.rglob("*") if d.is_dir() and any(x in d.name.lower() for x in ("mask","label","segment"))]
    if (root/"images").is_dir() and (root/"masks").is_dir(): image_dirs,mask_dirs=[root/"images"],[root/"masks"]
    pairs=[]; missing=0
    for image_dir in image_dirs:
        for mask_dir in [m for m in mask_dirs if m.parent==image_dir.parent]:
            masks={p.stem:p for p in mask_dir.rglob("*") if p.suffix.lower() in SUFFIXES}
            for image in sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in SUFFIXES):
                mask=masks.get(image.stem)
                if mask is None: missing+=1
                else: pairs.append(Pair(image,mask,_group(image)))
    seen=set(); unique=[]
    for p in pairs:
        key=(p.image.resolve(),p.mask.resolve())
        if key not in seen: seen.add(key); unique.append(p)
    return sorted(unique,key=lambda p:str(p.image)), {"root":str(root),"pairs":len(unique),"missing_masks":missing,"duplicate_pairs":len(pairs)-len(unique)}
def load_mask(path, shape=None):
    mask=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if mask is None: raise ValueError(f"Cannot decode mask: {path}")
    if shape and mask.shape!=shape: mask=cv2.resize(mask,(shape[1],shape[0]),interpolation=cv2.INTER_NEAREST)
    return (mask>0).astype(np.float32)
def split_pairs(pairs:Iterable[Pair],val_fraction=.15,test_fraction=.15,seed=42):
    pairs=list(pairs); groups=sorted({p.group for p in pairs}); rng=np.random.RandomState(seed); rng.shuffle(groups)
    nt=max(1,round(len(groups)*test_fraction)) if groups else 0; nv=max(1,round(len(groups)*val_fraction)) if groups else 0
    test,val=set(groups[:nt]),set(groups[nt:nt+nv]); train=set(groups[nt+nv:])
    return tuple([p for p in pairs if p.group in g] for g in (train,val,test))
