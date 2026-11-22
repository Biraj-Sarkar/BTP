"""Evaluate a saved segmentation checkpoint on a supplied dataset."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .dataset import discover_pairs, split_pairs
from .models import build_model
from .preprocessing import prepare
from .train import PairDataset
from .utils import metrics
def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--data',type=Path,required=True); p.add_argument('--device',default='cpu'); a=p.parse_args(); b=torch.load(a.checkpoint,map_location='cpu',weights_only=False); c=b.get('config',{}); m=build_model(b['architecture'],c.get('encoder','efficientnet-b0'),False); m.load_state_dict(b['state_dict']); m.to(a.device).eval(); pairs,_=discover_pairs(a.data); _,_,test=split_pairs(pairs,c.get('val_fraction',.15),c.get('test_fraction',.15),c.get('seed',42)); rows=[]; agg=[]
    with torch.no_grad():
      for x,y,ip,_ in DataLoader(PairDataset(test,c.get('size',320),False),batch_size=1):
        z=m(x.to(a.device)); mm=metrics(z.cpu(),y); rows.append({'image':ip[0],**mm}); agg.append(mm)
    if not agg: raise RuntimeError('No held-out samples found')
    summary={k:sum(x[k] for x in agg)/len(agg) for k in ('dice','iou','precision','recall','specificity','accuracy')}
    print(summary)
    with (a.checkpoint.parent/'test_predictions.csv').open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    (a.checkpoint.parent/'test_metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
if __name__=='__main__': main()
