"""Small training-only overfit check; never reads validation or test samples."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import torch
from .dataset import discover_pairs,split_pairs
from .models import build_model
from .preprocessing import prepare
from .train import loss_components

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True); p.add_argument('--model',required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--size',type=int,default=224); p.add_argument('--pretrained',action='store_true'); p.add_argument('--steps',type=int,default=10); p.add_argument('--seed',type=int,default=42); p.add_argument('--device',default='cpu'); a=p.parse_args(); torch.manual_seed(a.seed)
    pairs,_=discover_pairs(a.data); train,_,_=split_pairs(pairs,.15,.15,a.seed); xs=[]; ys=[]
    for pair in train[:2]: x,y=prepare(pair.image,pair.mask,a.size,False); xs.append(x); ys.append(y)
    x=torch.stack(xs).to(a.device); y=torch.stack(ys).to(a.device); model=build_model(a.model,'efficientnet-b0',a.pretrained).to(a.device); opt=torch.optim.AdamW(model.parameters(),lr=1e-3); model.train(); first=None; last=None
    for _ in range(a.steps):
        opt.zero_grad(); logits=model(x); loss,_=loss_components(logits,y,SimpleNamespace(loss='focal_dice',focal_alpha=.75,focal_gamma=2.,pos_weight=5.,tversky_alpha=.7,tversky_beta=.3)); loss.backward(); opt.step();
        with torch.no_grad(): pred=torch.sigmoid(logits)>=.5; true=y>=.5; dice=float((2*(pred&true).sum()+1e-6)/(pred.sum()+true.sum()+1e-6))
        first=float(loss) if first is None else first; last={'loss':float(loss),'dice':dice}
    result={'model':a.model,'pretrained':a.pretrained,'size':a.size,'steps':a.steps,'first_loss':first,'last_loss':last['loss'],'last_dice':last['dice'],'output_shape':list(logits.shape),'finite_logits':bool(torch.isfinite(logits).all())}; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result))
if __name__=='__main__': main()
