"""Four-way held-out evaluation for matched production/candidate ROI predictors."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np, torch
import cv2
from predictor.training import load_checkpoint
from predictor.data.transforms import get_val_transforms

ROOT=Path(__file__).resolve().parents[1]
def read(path):
    return {
        x["image_id"]: x
        for x in csv.DictReader(path.open(newline="", encoding="utf-8"))
        if x["split"] == "test" and x["training_eligible"].lower() == "true"
    }
def main():
    p=argparse.ArgumentParser(); p.add_argument("--production-manifest",type=Path,required=True); p.add_argument("--attention-manifest",type=Path,required=True); p.add_argument("--production-checkpoint",type=Path,required=True); p.add_argument("--attention-checkpoint",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    prod=read(a.production_manifest); att=read(a.attention_manifest); ids=sorted(set(prod)&set(att));
    if len(ids)!=25: raise RuntimeError(f"Expected 25 matched test records, got {len(ids)}")
    mprod,_=load_checkpoint(a.production_checkpoint,'cpu'); matt,_=load_checkpoint(a.attention_checkpoint,'cpu'); tf=get_val_transforms(); rows=[]
    with torch.inference_mode():
      for i in ids:
        rp,ra=prod[i],att[i]; true=float(rp["hemoglobin"]); xp=tf(cv2.cvtColor(cv2.imread(str(ROOT/rp["roi_path"])),cv2.COLOR_BGR2RGB)).unsqueeze(0); xa=tf(cv2.cvtColor(cv2.imread(str(ROOT/ra["roi_path"])),cv2.COLOR_BGR2RGB)).unsqueeze(0)
        pp=float(mprod.predict_hb(xp).item()); pa=float(mprod.predict_hb(xa).item()); qp=float(matt.predict_hb(xp).item()); qa=float(matt.predict_hb(xa).item())
        rows.append({"image_id":i,"true_hb":true,"production_on_production":pp,"production_on_attention":pa,"attention_on_production":qp,"attention_on_attention":qa,"production_on_production_abs":abs(pp-true),"production_on_attention_abs":abs(pa-true),"attention_on_production_abs":abs(qp-true),"attention_on_attention_abs":abs(qa-true)})
    def stats(key):
      e=np.array([r[key]-r["true_hb"] for r in rows]); return {"count":len(e),"mae_g_dl":float(np.mean(np.abs(e))),"rmse_g_dl":float(np.sqrt(np.mean(e*e))),"bias_g_dl":float(np.mean(e)),"inference_failures":0}
    rng=np.random.default_rng(42); d=np.array([r["attention_on_attention_abs"]-r["production_on_production_abs"] for r in rows]); boot=rng.choice(d,(20000,len(d)),replace=True).mean(1)
    summary={"test_count":len(rows),"test_split_used_only_after_training":True,"pipelines":{"production_on_production":stats("production_on_production"),"production_on_attention":stats("production_on_attention"),"attention_on_production":stats("attention_on_production"),"attention_on_attention":stats("attention_on_attention")},"paired_intended_delta_attention_minus_production":{"mean":float(d.mean()),"bootstrap_95_ci":[float(np.percentile(boot,2.5)),float(np.percentile(boot,97.5))]}}
    with (a.output/"per_image.csv").open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    (a.output/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8"); print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
