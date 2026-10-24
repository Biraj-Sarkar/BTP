"""Train one segmentation architecture with validation-only selection."""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from .dataset import discover_pairs, split_pairs
from .models import build_model
from .preprocessing import prepare
from .utils import metrics, seed_everything, save_checkpoint, json_dump

class PairDataset(Dataset):
    def __init__(self,pairs,size,train=False,seed=42): self.pairs=list(pairs); self.size=size; self.train=train; self.rng=__import__('numpy').random.RandomState(seed)
    def __len__(self): return len(self.pairs)
    def __getitem__(self,i):
        x,y=prepare(self.pairs[i].image,self.pairs[i].mask,self.size,self.train,self.rng); return x,y,str(self.pairs[i].image),self.pairs[i].group

def loss_components(logits, target, args):
    p = torch.sigmoid(logits); dims=(1,2,3)
    soft_dice = 1 - (2*(p*target).sum(dims)+1) / (p.sum(dims)+target.sum(dims)+1)
    if args.loss == 'bce_dice':
        bce = F.binary_cross_entropy_with_logits(logits, target); primary = bce
    elif args.loss == 'weighted_bce_dice':
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=torch.tensor(args.pos_weight, device=logits.device)); primary = bce
    elif args.loss == 'focal_dice':
        bce_raw = F.binary_cross_entropy_with_logits(logits, target, reduction='none'); pt=torch.where(target > .5, p, 1-p); primary=(args.focal_alpha*((1-pt).clamp_min(1e-6)**args.focal_gamma)*bce_raw).mean(); bce=primary
    elif args.loss == 'tversky':
        tp=(p*target).sum(dims); fp=(p*(1-target)).sum(dims); fn=((1-p)*target).sum(dims); primary=(1-(tp+1)/(tp+args.tversky_alpha*fp+args.tversky_beta*fn+1)).mean(); bce=primary
    else: raise ValueError(args.loss)
    total = primary + soft_dice.mean()
    return total, {'primary_loss': float(primary.detach().item()), 'dice_loss': float(soft_dice.mean().detach().item())}

def loss_fn(logits,target,args): return loss_components(logits,target,args)[0]

def run(args):
    seed_everything(args.seed); pairs,audit=discover_pairs(args.data)
    if not pairs: raise RuntimeError(f"No valid image/mask pairs found under {args.data}; refusing to train on predicted ROIs.")
    train,val,test=split_pairs(pairs,args.val_fraction,args.test_fraction,args.seed); root=Path(args.output)/(args.experiment or ''); out=root/args.model.replace('+','plus'); out.mkdir(parents=True,exist_ok=True)
    manifest=root/'split_manifest.csv'; root.mkdir(parents=True,exist_ok=True)
    if not manifest.exists():
        with manifest.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=['image','mask','group','split']); w.writeheader(); w.writerows({'image':str(p.image),'mask':str(p.mask),'group':p.group,'split':s} for s,items in (('train',train),('val',val),('test',test)) for p in items)
    model=build_model(args.model,args.encoder,args.pretrained); device=torch.device(args.device); model.to(device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay); best=-1; stale=0; history=[]
    loaders=[DataLoader(PairDataset(p,args.size,t,args.seed),batch_size=args.batch_size,shuffle=t,num_workers=0) for p,t in ((train,True),(val,False),(test,False))]
    for epoch in range(1,args.epochs+1):
        model.train(); tr=0; train_primary=0; train_dice=0; train_pf=0
        for x,y,_,_ in loaders[0]: opt.zero_grad(); z=model(x.to(device)); l, parts=loss_components(z,y.to(device),args); l.backward(); opt.step(); tr+=l.item(); train_primary+=parts['primary_loss']; train_dice+=parts['dice_loss']; train_pf+=(torch.sigmoid(z)>=args.metric_threshold).float().mean().item()
        model.eval(); vm=[]; vl=0; val_primary=0; val_dice_loss=0; val_pf=0
        with torch.no_grad():
            for x,y,_,_ in loaders[1]:
                z=model(x.to(device)); l, parts=loss_components(z,y.to(device),args); vl+=l.item(); val_primary+=parts['primary_loss']; val_dice_loss+=parts['dice_loss']; val_pf+=(torch.sigmoid(z)>=args.metric_threshold).float().mean().item()
                vm.extend(metrics(zi.unsqueeze(0).cpu(), yi.unsqueeze(0).cpu(), args.metric_threshold) for zi,yi in zip(z,y.to(device)))
        valm={k:sum(m[k] for m in vm)/len(vm) for k in vm[0]}; history.append({"epoch":epoch,"train_loss":tr/max(1,len(loaders[0])),"val_loss":vl/max(1,len(loaders[1])),"train_primary_loss":train_primary/max(1,len(loaders[0])),"train_dice_loss":train_dice/max(1,len(loaders[0])),"val_primary_loss":val_primary/max(1,len(loaders[1])),"val_dice_loss":val_dice_loss/max(1,len(loaders[1])),"train_pred_fraction":train_pf/max(1,len(loaders[0])),"val_pred_fraction":val_pf/max(1,len(loaders[1])),**valm})
        if valm["dice"]>best: best=valm["dice"]; stale=0; save_checkpoint(out/'best_checkpoint.pt',model,args.model,vars(args),epoch,valm)
        else: stale+=1
        if stale>=args.patience: break
    json_dump(out/'audit.json',{**audit,"train":len(train),"val":len(val),"test":len(test)})
    (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    print(f"{args.model}: pairs={len(pairs)}, train={len(train)}, val={len(val)}, test={len(test)}, best_val_dice={best:.4f}")

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True); p.add_argument('--model',choices=['unet','unet++','attention_unet'],required=True); p.add_argument('--output',default='extractor/experiments'); p.add_argument('--experiment',default=''); p.add_argument('--encoder',default='efficientnet-b0'); p.add_argument('--pretrained',action='store_true'); p.add_argument('--size',type=int,default=320); p.add_argument('--epochs',type=int,default=50); p.add_argument('--batch-size',type=int,default=8); p.add_argument('--lr',type=float,default=1e-4); p.add_argument('--weight-decay',type=float,default=1e-4); p.add_argument('--patience',type=int,default=8); p.add_argument('--val-fraction',type=float,default=.15); p.add_argument('--test-fraction',type=float,default=.15); p.add_argument('--seed',type=int,default=42); p.add_argument('--loss',choices=['bce_dice','weighted_bce_dice','focal_dice','tversky'],default='bce_dice'); p.add_argument('--pos-weight',type=float,default=5.0); p.add_argument('--focal-alpha',type=float,default=.75); p.add_argument('--focal-gamma',type=float,default=2.0); p.add_argument('--tversky-alpha',type=float,default=.7); p.add_argument('--tversky-beta',type=float,default=.3); p.add_argument('--metric-threshold',type=float,default=.5); p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); run(p.parse_args())
