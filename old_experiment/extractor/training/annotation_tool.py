"""Interactive OpenCV review tool with aspect-correct mouse mapping."""
from __future__ import annotations
import argparse, csv
from pathlib import Path
from typing import Optional, Tuple
import cv2, numpy as np

CANVAS_SIZE = (900, 600)  # width, height

def fit_geometry(image_shape: Tuple[int, int], canvas_size: Tuple[int, int] = CANVAS_SIZE):
    image_h, image_w = image_shape[:2]; canvas_w, canvas_h = canvas_size
    scale = min(canvas_w / image_w, canvas_h / image_h)
    display_w = max(1, int(round(image_w * scale))); display_h = max(1, int(round(image_h * scale)))
    return scale, (canvas_w - display_w) // 2, (canvas_h - display_h) // 2, display_w, display_h

def display_to_mask(x: int, y: int, mask_shape: Tuple[int, int], geometry) -> Optional[Tuple[int, int]]:
    scale, offset_x, offset_y, display_w, display_h = geometry
    if x < offset_x or y < offset_y or x >= offset_x + display_w or y >= offset_y + display_h: return None
    mask_h, mask_w = mask_shape[:2]
    return (min(mask_w - 1, max(0, int((x - offset_x) * mask_w / display_w))), min(mask_h - 1, max(0, int((y - offset_y) * mask_h / display_h))))

def _persist(path: Path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path('segmentation_dataset')); p.add_argument('--image-id',default=None,help='Open this image first'); a=p.parse_args(); mf=a.root/'annotation_manifest.csv'; rows=list(csv.DictReader(mf.open(encoding='utf-8'))); idx=next((i for i,r in enumerate(rows) if r['image_id']==a.image_id),0) if a.image_id else 0
    while idx<len(rows):
        r=rows[idx]; image=cv2.imread(str(a.root/r['image_path'])); candidate=cv2.imread(str(a.root/r['candidate_path']),cv2.IMREAD_GRAYSCALE); existing=a.root/r['mask_path'] if r.get('mask_path') else None; saved=cv2.imread(str(existing),cv2.IMREAD_GRAYSCALE) if existing and existing.is_file() else None; mask=saved.copy() if saved is not None else candidate.copy(); geometry=fit_geometry(image.shape[:2]); _,ox,oy,dw,dh=geometry; cursor=None; window='Annotate conjunctiva (n save, a candidate, c clear, q quit)'
        def draw(event,x,y,flags,param):
            nonlocal cursor
            mapped=display_to_mask(x,y,mask.shape,geometry); cursor=(x,y) if mapped else None
            if mapped is None: return
            radius=max(1,int(round(6/geometry[0]))); mx,my=mapped
            if event==cv2.EVENT_LBUTTONDOWN or (event==cv2.EVENT_MOUSEMOVE and flags&cv2.EVENT_FLAG_LBUTTON): cv2.circle(mask,(mx,my),radius,255,-1)
            elif event==cv2.EVENT_RBUTTONDOWN or (event==cv2.EVENT_MOUSEMOVE and flags&cv2.EVENT_FLAG_RBUTTON): cv2.circle(mask,(mx,my),radius,0,-1)
        cv2.namedWindow(window); cv2.setMouseCallback(window,draw)
        while True:
            canvas=np.zeros((CANVAS_SIZE[1],CANVAS_SIZE[0],3),dtype=np.uint8); shown=cv2.resize(image,(dw,dh),interpolation=cv2.INTER_AREA); shown_mask=cv2.resize(mask,(dw,dh),interpolation=cv2.INTER_NEAREST); shown[shown_mask>0]=(0.55*shown[shown_mask>0]+0.45*np.array([0,0,255])).astype('uint8'); canvas[oy:oy+dh,ox:ox+dw]=shown
            if cursor is not None:
                cv2.drawMarker(canvas,cursor,(0,255,255),cv2.MARKER_CROSS,16,1); mapped=display_to_mask(cursor[0],cursor[1],mask.shape,geometry); cv2.putText(canvas,f'display={cursor} mask={mapped}',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),1,cv2.LINE_AA)
            cv2.imshow(window,canvas); key=cv2.waitKey(20)&255
            if key==ord('a'): mask=candidate.copy()
            elif key==ord('c'): mask=np.zeros_like(mask)
            elif key in (ord('n'),32):
                out=a.root/'masks'/f"{r['image_id']}.png"; out.parent.mkdir(parents=True,exist_ok=True); cv2.imwrite(str(out),mask); r['mask_path']=str(out.relative_to(a.root)); r['annotation_status']='reviewed_accepted' if np.array_equal(mask,candidate) else 'reviewed_corrected'; r['reviewed']='true'; r['validation_status']='pending'; _persist(mf,rows); idx+=1; break
            elif key==ord('q'): cv2.destroyAllWindows(); return
        cv2.destroyWindow(window)
    _persist(mf,rows); print('All records saved; run validate_annotations.py before training.')
if __name__=='__main__': main()
