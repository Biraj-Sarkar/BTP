"""Reproducibility, metrics, and checkpoint helpers."""
import json,random,time
from pathlib import Path
import numpy as np, torch
def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
def metrics(logits,target,threshold=.5):
    pred=torch.sigmoid(logits)>=threshold; true=target>=.5; tp=(pred&true).sum().item(); tn=((~pred)&(~true)).sum().item(); fp=(pred&(~true)).sum().item(); fn=((~pred)&true).sum().item()
    def r(a,b): return a/b if b else (1. if a==0 else 0.)
    return {"dice":r(2*tp,2*tp+fp+fn),"iou":r(tp,tp+fp+fn),"precision":r(tp,tp+fp),"recall":r(tp,tp+fn),"specificity":r(tn,tn+fp),"accuracy":r(tp+tn,tp+tn+fp+fn),"tp":tp,"tn":tn,"fp":fp,"fn":fn}
def save_checkpoint(path,model,architecture,config,epoch,val_metrics):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); torch.save({"format":"anemia-segmenter","version":2,"architecture":architecture,"state_dict":model.state_dict(),"config":config,"epoch":epoch,"val_metrics":val_metrics,"created_at":time.time()},path)
def json_dump(path,value): Path(path).write_text(json.dumps(value,indent=2,default=float),encoding="utf-8")
