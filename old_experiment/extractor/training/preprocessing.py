"""Shared image/mask preprocessing used by all architectures."""
import cv2, numpy as np, torch
MEAN=np.array([.485,.456,.406],dtype=np.float32); STD=np.array([.229,.224,.225],dtype=np.float32)
def prepare(image_path,mask_path,size=320,train=False,rng=None):
    image=cv2.imread(str(image_path)); mask=cv2.imread(str(mask_path),cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None: raise ValueError(f"Unreadable image/mask: {image_path}, {mask_path}")
    image=cv2.cvtColor(cv2.resize(image,(size,size),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB); mask=cv2.resize(mask,(size,size),interpolation=cv2.INTER_NEAREST)
    if train and (rng or np.random).random()<.5: image,mask=image[:,::-1].copy(),mask[:,::-1].copy()
    x=(image.astype(np.float32)/255.-MEAN)/STD
    return torch.from_numpy(x.transpose(2,0,1)).float(),torch.from_numpy((mask>0).astype(np.float32)[None])
