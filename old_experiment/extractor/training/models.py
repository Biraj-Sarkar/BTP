"""Common one-channel segmentation models: U-Net, U-Net++, Attention U-Net."""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
def _smp():
    import segmentation_models_pytorch as smp
    return smp
class AttentionUnet(nn.Module):
    """Attention-gated decoder over an SMP encoder (Oktay et al., 2018)."""
    def __init__(self,encoder="efficientnet-b0",pretrained=False):
        super().__init__(); from segmentation_models_pytorch.encoders import get_encoder
        self.encoder=get_encoder(encoder,in_channels=3,depth=5,weights="imagenet" if pretrained else None); channels=list(self.encoder.out_channels)[::-1]; widths=[256,128,64,32,16]; self.blocks=nn.ModuleList(); self.gates=nn.ModuleList(); inp=channels[0]
        for skip,out in zip(channels[1:],widths):
            self.gates.append(nn.Sequential(nn.Conv2d(skip,out,1),nn.ReLU(),nn.Conv2d(out,1,1),nn.Sigmoid())); self.blocks.append(nn.Sequential(nn.Conv2d(inp+skip,out,3,padding=1),nn.BatchNorm2d(out),nn.ReLU(),nn.Conv2d(out,out,3,padding=1),nn.BatchNorm2d(out),nn.ReLU())); inp=out
        self.head=nn.Conv2d(widths[-1],1,1)
    def forward(self,x):
        feats=self.encoder(x)[::-1]; out=feats[0]
        for i,block in enumerate(self.blocks):
            skip=feats[i+1]; out=F.interpolate(out,size=skip.shape[-2:],mode="bilinear",align_corners=False); out=block(torch.cat([out,skip*self.gates[i](skip)],1))
        return self.head(out)
def build_model(name,encoder="efficientnet-b0",pretrained=False):
    smp=_smp(); weights="imagenet" if pretrained else None
    if name=="unet": return smp.Unet(encoder_name=encoder,encoder_weights=weights,in_channels=3,classes=1)
    if name in {"unet++","unetplusplus"}: return smp.UnetPlusPlus(encoder_name=encoder,encoder_weights=weights,in_channels=3,classes=1)
    if name in {"attention_unet","attention"}: return AttentionUnet(encoder,pretrained)
    raise ValueError(f"Unknown model {name!r}")
