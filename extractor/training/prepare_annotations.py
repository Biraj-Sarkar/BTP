"""Create annotation-ready copies and candidate masks; never marks them as ground truth."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import cv2
from extractor import ROIExtractor

FIELDS = ["image_id","image_path","mask_path","candidate_path","width","height","mask_width","mask_height","candidate_source","annotation_status","reviewed","reviewer","review_notes","validation_status"]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--images',type=Path,default=Path('datasets')); p.add_argument('--output',type=Path,default=Path('segmentation_dataset')); p.add_argument('--checkpoint',type=Path,default=Path('Models/UNet_extractor/unet.pt')); a=p.parse_args()
    images=sorted(x for x in a.images.iterdir() if x.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'})
    if not images: raise RuntimeError(f'No images found in {a.images}')
    for d in ('images','candidate_masks','review','review/overlays','masks'): (a.output/d).mkdir(parents=True,exist_ok=True)
    extractor=ROIExtractor(checkpoint_path=a.checkpoint); rows=[]
    for i,path in enumerate(images,1):
        image=cv2.imread(str(path),cv2.IMREAD_COLOR)
        if image is None: raise RuntimeError(f'Unreadable image: {path}')
        h,w=image.shape[:2]; image_id=path.stem
        copied=a.output/'images'/path.name; cv2.imwrite(str(copied),image)
        rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB); result=extractor.extract(rgb)
        candidate=cv2.resize(result.mask,(w,h),interpolation=cv2.INTER_NEAREST)
        candidate_path=a.output/'candidate_masks'/f'{image_id}.png'; cv2.imwrite(str(candidate_path),candidate)
        overlay=rgb.copy(); overlay[candidate>0]=(0.55*overlay[candidate>0]+0.45*__import__('numpy').array([255,0,0])).astype('uint8')
        cv2.imwrite(str(a.output/'review'/'overlays'/f'{image_id}.jpg'),cv2.cvtColor(overlay,cv2.COLOR_RGB2BGR))
        rows.append({"image_id":image_id,"image_path":str(copied.relative_to(a.output)),"mask_path":"","candidate_path":str(candidate_path.relative_to(a.output)),"width":w,"height":h,"mask_width":w,"mask_height":h,"candidate_source":"existing_unet_inference","annotation_status":"candidate_generated","reviewed":"false","reviewer":"","review_notes":"","validation_status":"pending"})
        if i%20==0 or i==len(images): print(f'Generated {i}/{len(images)} candidates')
    with (a.output/'annotation_manifest.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
    (a.output/'README.md').write_text('# Segmentation annotation dataset\n\nOriginal images are copied under `images/`. Candidate masks under `candidate_masks/` are generated suggestions only and are not ground truth. Use `annotation_tool.py` to review/correct each image and save reviewed masks under `masks/`. Training refuses rows without reviewed final masks.\n',encoding='utf-8')
    print(f'Created {len(rows)} candidate records at {a.output}; human review is still required.')
if __name__=='__main__': main()
