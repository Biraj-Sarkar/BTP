"""Validate final human-reviewed masks and enforce the training gate."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import cv2,numpy as np
def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path('segmentation_dataset')); a=p.parse_args(); manifest=a.root/'annotation_manifest.csv'
    if not manifest.is_file(): raise FileNotFoundError(manifest)
    rows=list(csv.DictReader(manifest.open(encoding='utf-8'))); report=[]; eligible=0
    for r in rows:
        errors=[]; image_path=a.root/r['image_path']; mask_path=a.root/r['mask_path'] if r.get('mask_path') else None
        image=cv2.imread(str(image_path),cv2.IMREAD_COLOR) if image_path.is_file() else None; mask=cv2.imread(str(mask_path),cv2.IMREAD_GRAYSCALE) if mask_path and mask_path.is_file() else None
        if image is None: errors.append('missing_or_unreadable_image')
        if mask is None: errors.append('missing_or_unreadable_final_mask')
        if mask is not None and image is not None:
            if mask.shape!=image.shape[:2]: errors.append('dimension_mismatch')
            values=set(np.unique(mask).tolist())
            if not values.issubset({0,255}): errors.append('non_binary_mask')
            area=float(np.count_nonzero(mask))/mask.size
            if area==0: errors.append('empty_mask')
            if area<.0005 or area>.85: errors.append('suspicious_foreground_area')
        if str(r.get('reviewed','')).lower()!='true': errors.append('not_human_reviewed')
        status='eligible' if not errors else 'invalid_or_pending'; eligible += status=='eligible'
        report.append({"image_id":r.get('image_id',''),"status":status,"errors":";".join(errors)})
    with (a.root/'validation_report.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['image_id','status','errors']); w.writeheader(); w.writerows(report)
    summary={'total':len(rows),'eligible':eligible,'pending_or_invalid':len(rows)-eligible,'training_allowed':eligible==len(rows) and len(rows)>0}
    (a.root/'validation_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2))
    if not summary['training_allowed']: raise SystemExit('Training blocked: every sample must have a valid, human-reviewed final mask.')
if __name__=='__main__': main()
