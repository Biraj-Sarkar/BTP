"""Train a predictor from an authoritative fixed-split ROI manifest."""
from __future__ import annotations
import argparse, csv, random
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from predictor.models import build_model
from predictor.data import HbDataset, get_train_transforms, get_val_transforms
from predictor.data.manifest import SampleRecord
from predictor.training import Trainer

ROOT = Path(__file__).resolve().parents[1]

def load_rows(path: Path):
    rows=[]
    with path.open(newline="",encoding="utf-8") as f:
        for x in csv.DictReader(f):
            if x.get("training_eligible", "false").lower() != "true": continue
            roi=ROOT / x["roi_path"]
            if not roi.is_file() or x["roi_shape"] != "224x224x3" or x["roi_dtype"] != "uint8":
                raise RuntimeError(f"Invalid usable ROI row: {x['image_id']}")
            rows.append(SampleRecord(patient_id=x["image_id"], image_id=x["image_id"], image_path=roi, hb=float(x["hemoglobin"]), age=float(x["age"]) if x["age"] else None, sex=x["sex"], usable=True, mask_ratio=float(x["mask_ratio"]) if x["mask_ratio"] else None))
    return rows

def main():
    p=argparse.ArgumentParser(); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--paired-manifest",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--seed",type=int,default=42); p.add_argument("--epochs",type=int,default=8); p.add_argument("--batch-size",type=int,default=16); p.add_argument("--lr",type=float,default=3e-4); p.add_argument("--weight-decay",type=float,default=1e-4); p.add_argument("--patience",type=int,default=3); a=p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    rows=load_rows(a.manifest)
    paired_ids={x["image_id"] for x in csv.DictReader(a.paired_manifest.open(newline="",encoding="utf-8")) if x.get("training_eligible", "false").lower()=="true"}
    rows=[r for r in rows if r.image_id in paired_ids]
    # The split is authoritative in the manifest; use a parallel lookup from CSV rows.
    raw=list(csv.DictReader(a.manifest.open(newline="",encoding="utf-8"))); split_by={x["image_id"]:x["split"] for x in raw}; groups={s:[] for s in ("train","val","test")}
    for r in rows: groups[split_by[r.image_id]].append(r)
    counts={s:len(groups[s]) for s in groups}; expected={"train":122,"val":27,"test":25}
    if counts != expected: raise RuntimeError(f"Unexpected fixed split counts {counts}; expected {expected}")
    train,val,test=groups["train"],groups["val"],groups["test"]
    target_mean=float(np.mean([r.hb for r in train])); target_std=float(np.std([r.hb for r in train]));
    cfg={"architecture":"resnet","variant":"resnet18","pretrained":True,"dropout":0.3,"epochs":a.epochs,"batch_size":a.batch_size,"learning_rate":a.lr,"weight_decay":a.weight_decay,"loss_type":"smooth_l1","severe_hb_weight":1.2,"patience":a.patience,"seed":a.seed,"fixed_split_manifest":str(a.manifest),"split_counts":counts,"target_mean":target_mean,"target_std":target_std,"filter_usable":True}
    train_ds=HbDataset(train,get_train_transforms(),target_mean,target_std); val_ds=HbDataset(val,get_val_transforms(),target_mean,target_std); test_ds=HbDataset(test,get_val_transforms(),target_mean,target_std)
    loaders=[DataLoader(train_ds,batch_size=a.batch_size,shuffle=True),DataLoader(val_ds,batch_size=a.batch_size,shuffle=False),DataLoader(test_ds,batch_size=a.batch_size,shuffle=False)]
    model=build_model("resnet",variant="resnet18",pretrained=True,dropout=.3,target_mean=target_mean,target_std=target_std)
    audit={"fixed_split":True,"source_manifest":str(a.manifest),"counts":counts,"excluded_rows":183-len(rows),"patient_ids_available":False}
    summary=Trainer(model,*loaders,config=cfg,output_dir=ROOT/a.output,audit_summary=audit).run(); print(summary["test_metrics"])
if __name__=="__main__": main()
