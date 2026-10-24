"""Select a segmentation threshold on validation data only."""
import argparse,csv,json
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from .dataset import discover_pairs, split_pairs
from .models import build_model
from .train import PairDataset
from .utils import metrics

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--data',type=Path,required=True); p.add_argument('--output',type=Path); p.add_argument('--device',default='cpu'); p.add_argument('--start',type=float,default=.1); p.add_argument('--stop',type=float,default=.9); p.add_argument('--step',type=float,default=.05); a=p.parse_args()
    b=torch.load(a.checkpoint,map_location='cpu',weights_only=False); c=b.get('config',{}); m=build_model(b['architecture'],c.get('encoder','efficientnet-b0'),False); m.load_state_dict(b['state_dict']); m.to(a.device).eval(); pairs,_=discover_pairs(a.data); _,val,_=split_pairs(pairs,c.get('val_fraction',.15),c.get('test_fraction',.15),c.get('seed',42)); probs=[]; targets=[]
    with torch.inference_mode():
        for x,y,_,_ in DataLoader(PairDataset(val,c.get('size',320),False),batch_size=8): probs.append(torch.sigmoid(m(x.to(a.device))).cpu()); targets.append(y)
    probs=torch.cat(probs); targets=torch.cat(targets); rows=[]; per_image=[]
    for t in np.arange(a.start,a.stop+a.step/2,a.step):
        z=[]
        for logit,target in zip(torch.logit(probs.clamp(1e-6,1-1e-6)),targets): z.append(metrics(logit[None],target[None],float(t)))
        row={k:float(np.mean([x[k] for x in z])) for k in ('dice','iou','precision','recall','specificity','accuracy','tp','tn','fp','fn')}; row['threshold']=float(t); row['mean_pred_fraction']=float(((probs>=t).float()).mean()); rows.append(row)
        per_image.extend({'image':str(val[i].image),'threshold':float(t),**z[i]} for i in range(len(z)))
    out=a.output or a.checkpoint.parent/'threshold_sweep.csv'; out.parent.mkdir(parents=True,exist_ok=True); out.write_text('');
    with out.open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    with (out.parent/'threshold_per_image.csv').open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=per_image[0].keys()); w.writeheader(); w.writerows(per_image)
    best=max(rows,key=lambda x:x['dice']); (out.parent/'selected_threshold.json').write_text(json.dumps({'checkpoint':str(a.checkpoint),'selection':'validation_dice','selected':best,'fixed_0.5':next(x for x in rows if abs(x['threshold']-.5)<1e-6)},indent=2),encoding='utf-8'); print(json.dumps({'selected':best,'fixed_0.5':next(x for x in rows if abs(x['threshold']-.5)<1e-6)},indent=2))
if __name__=='__main__': main()
