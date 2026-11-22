"""Create a read-only contact sheet of reviewed masks over original images."""
import argparse,csv,math
from pathlib import Path
import cv2,numpy as np

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path('segmentation_dataset')); p.add_argument('--output',type=Path,default=Path('extractor/experiments/annotation_audit/contact_sheet.jpg')); p.add_argument('--columns',type=int,default=12); p.add_argument('--tile-width',type=int,default=160); a=p.parse_args(); rows=list(csv.DictReader((a.root/'annotation_manifest.csv').open(encoding='utf-8'))); tiles=[]
    for r in rows:
        im=cv2.imread(str(a.root/r['image_path'])); m=cv2.imread(str(a.root/r['mask_path']),cv2.IMREAD_GRAYSCALE)
        if im is None or m is None: continue
        h=int(a.tile_width*im.shape[0]/im.shape[1]); im=cv2.resize(im,(a.tile_width,h),interpolation=cv2.INTER_AREA); m=cv2.resize(m,(a.tile_width,h),interpolation=cv2.INTER_NEAREST); overlay=im.copy(); overlay[m>0]=(0.55*overlay[m>0]+0.45*np.array([0,255,0])).astype(np.uint8); cv2.putText(overlay,r['image_id'],(3,16),cv2.FONT_HERSHEY_SIMPLEX,.38,(255,255,255),1,cv2.LINE_AA); tiles.append(overlay)
    tile_h=max(x.shape[0] for x in tiles); cols=a.columns; rows_n=math.ceil(len(tiles)/cols); canvas=np.zeros((rows_n*tile_h,cols*a.tile_width,3),np.uint8)
    for i,t in enumerate(tiles):
        x=(i%cols)*a.tile_width; y=(i//cols)*tile_h; canvas[y:y+t.shape[0],x:x+t.shape[1]]=t
    a.output.parent.mkdir(parents=True,exist_ok=True); cv2.imwrite(str(a.output),canvas); print({'images':len(tiles),'output':str(a.output),'shape':canvas.shape})
if __name__=='__main__': main()
