"""Self-contained Hugging Face backed extraction and Hb inference."""
from __future__ import annotations
import json, os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import cv2, numpy as np, torch
from torchvision.models import convnext_tiny, resnet18

def _asset(repo_env: str, file_env: str, default_file: str) -> Path:
    local = os.getenv(file_env)
    if local: return Path(local)
    repo = os.getenv(repo_env) or os.getenv("HF_MODEL_REPO")
    if not repo: raise RuntimeError(f"Set {file_env} or {repo_env}/HF_MODEL_REPO")
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=repo, filename=os.getenv(file_env + "_NAME", default_file)))

class UNetExtractor:
    def __init__(self, bundle_path: Path):
        import segmentation_models_pytorch as smp
        bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        config = bundle["config"]
        self.size = int(config.get("input_size", 320)); self.threshold = float(config.get("threshold", .5))
        self.mean = np.asarray(config.get("mean", (.485, .456, .406)), np.float32)
        self.std = np.asarray(config.get("std", (.229, .224, .225)), np.float32)
        self.min_area = float(config.get("min_area_fraction", .002))
        self.model = smp.Unet(encoder_name=config.get("encoder", "efficientnet-b0"), encoder_weights=None, in_channels=3, classes=1)
        self.model.load_state_dict(bundle["state_dict"]); self.model.eval()
    def __call__(self, image_rgb: np.ndarray) -> np.ndarray:
        h, w = image_rgb.shape[:2]
        x = cv2.resize(image_rgb, (self.size, self.size)).astype(np.float32) / 255.
        x = (x - self.mean) / self.std
        with torch.inference_mode(): p = torch.sigmoid(self.model(torch.from_numpy(x.transpose(2,0,1)).unsqueeze(0)))[0,0].numpy()
        mask = (cv2.resize(p, (w,h)) >= self.threshold).astype(np.uint8) * 255
        if not mask.any(): return mask
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if count > 1: mask = np.where(labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]), 255, 0).astype(np.uint8)
        flood = mask.copy(); border = np.zeros((h+2,w+2), np.uint8); cv2.floodFill(flood, border, (0,0), 255)
        mask = mask | cv2.bitwise_not(flood)
        return mask if np.count_nonzero(mask) / mask.size >= self.min_area else np.zeros_like(mask)

class AnubhavRegressor(torch.nn.Module):
    def __init__(self, checkpoint: dict):
        super().__init__(); net = resnet18(weights=None); net.fc = torch.nn.Sequential(torch.nn.Dropout(.3), torch.nn.Linear(net.fc.in_features,1)); self.net = net
        self.register_buffer("norm_mean", torch.tensor((.485,.456,.406)).view(1,3,1,1)); self.register_buffer("norm_std", torch.tensor((.229,.224,.225)).view(1,3,1,1))
        self.load_state_dict(checkpoint["state_dict"]); self.eval(); self.target_mean=float(checkpoint["target_mean"]); self.target_std=float(checkpoint["target_std"])
    def forward(self, x): return self.net((x-self.norm_mean)/self.norm_std)

class HBPredictor(torch.nn.Module):
    def __init__(self, num_sexes: int, state: dict):
        super().__init__(); b=convnext_tiny(weights=None); f=b.classifier[2].in_features; b.classifier=torch.nn.Identity(); self.backbone=b
        self.image_projection=torch.nn.Sequential(torch.nn.Linear(f,256),torch.nn.ReLU(),torch.nn.Dropout(.25)); self.age_encoder=torch.nn.Sequential(torch.nn.Linear(1,32),torch.nn.ReLU(),torch.nn.Linear(32,32),torch.nn.ReLU()); self.sex_embedding=torch.nn.Embedding(num_sexes,16); self.demographic_encoder=torch.nn.Sequential(torch.nn.Linear(48,64),torch.nn.ReLU(),torch.nn.Dropout(.125)); self.regressor=torch.nn.Sequential(torch.nn.Linear(320,128),torch.nn.ReLU(),torch.nn.Dropout(.25),torch.nn.Linear(128,32),torch.nn.ReLU(),torch.nn.Linear(32,1)); self.load_state_dict(state); self.eval()
    def forward(self,image,age,sex):
        x=torch.flatten(self.backbone.avgpool(self.backbone.features(image)),1); image=self.image_projection(x); demo=self.demographic_encoder(torch.cat((self.age_encoder(age),self.sex_embedding(sex)),1)); return self.regressor(torch.cat((image,demo),1)).squeeze(1)

@dataclass
class BackendModels:
    extractor: UNetExtractor; anubhav: Optional[AnubhavRegressor]; hb_predictor: Optional[HBPredictor]; age_mean: float; age_std: float; sex_to_index: dict
    @classmethod
    def load(cls):
        unet=UNetExtractor(_asset("HF_UNET_REPO","HF_UNET_PATH","unet.pt")); anubhav=None; hb=None
        try: anubhav=AnubhavRegressor(torch.load(_asset("HF_ANUBHAV_REPO","HF_ANUBHAV_PATH","best.pt"),map_location="cpu",weights_only=False))
        except (RuntimeError,ValueError,FileNotFoundError): pass
        try:
            blob=torch.load(_asset("HF_HB_REPO","HF_HB_PATH","best_model.pt"),map_location="cpu",weights_only=False); stats=json.loads(_asset("HF_HB_REPO","HF_HB_STATS_PATH","data_statistics.json").read_text()); hb=HBPredictor(int(blob["num_sexes"]),blob["model_state"]); return cls(unet,anubhav,hb,float(stats["age_mean"]),float(stats["age_std"]),stats["sex_to_index"])
        except (RuntimeError,ValueError,FileNotFoundError): pass
        return cls(unet,anubhav,hb,43.95,16.21,{"f":0,"m":1})
    def predict(self,image_rgb,age,sex):
        work=cv2.resize(image_rgb,(512,512)); means=work.reshape(-1,3).mean(0); balanced=np.clip(work*(means.mean()/np.maximum(means,1e-6)),0,255).astype(np.uint8); mask=self.extractor(balanced)
        if not mask.any(): raise ValueError("U-Net could not find a usable conjunctiva ROI")
        ys,xs=np.where(mask>0); pad=int(.08*max(xs.max()-xs.min(),ys.max()-ys.min())); x0,x1=max(0,xs.min()-pad),min(512,xs.max()+pad+1); y0,y1=max(0,ys.min()-pad),min(512,ys.max()+pad+1); crop=cv2.resize(np.where(mask[y0:y1,x0:x1,None]>0,balanced[y0:y1,x0:x1],0),(224,224)); tensor=torch.from_numpy(crop.transpose(2,0,1)).float().unsqueeze(0)/255.; result={"roi_backend":"unet","bbox":[int(x0),int(y0),int(x1),int(y1)]}
        with torch.inference_mode():
            if self.anubhav is not None: result.update(hb_g_dl=round(float(self.anubhav(tensor).item()*self.anubhav.target_std+self.anubhav.target_mean),2),model="anubhav")
            elif self.hb_predictor is not None:
                key=sex.strip().lower()[0]; age_t=torch.tensor([[(age-self.age_mean)/max(self.age_std,1e-6)]]); sex_t=torch.tensor([self.sex_to_index.get(key,0)]); image=(tensor-torch.tensor((.485,.456,.406)).view(1,3,1,1))/torch.tensor((.229,.224,.225)).view(1,3,1,1); result.update(hb_g_dl=round(float(self.hb_predictor(image,age_t,sex_t).item()),2),model="hbpredictor")
            else: raise RuntimeError("No Hb checkpoint could be loaded")
        threshold=13.0 if sex.strip().lower().startswith("m") else 12.0; result.update(anemic=result["hb_g_dl"]<threshold,threshold_g_dl=threshold); return result
