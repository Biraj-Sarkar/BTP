from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


class HbPredictor(nn.Module):
    """ConvNeXt-Tiny ROI encoder fused with age and sex embeddings."""
    def __init__(self, num_sexes: int, pretrained: bool = True, dropout: float = 0.25):
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        backbone = convnext_tiny(weights=weights)
        image_features = backbone.classifier[2].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.image_projection = nn.Sequential(nn.Linear(image_features, 256), nn.ReLU(), nn.Dropout(dropout))
        self.age_encoder = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.sex_embedding = nn.Embedding(num_sexes, 16)
        self.demographic_encoder = nn.Sequential(nn.Linear(48, 64), nn.ReLU(), nn.Dropout(dropout / 2))
        self.regressor = nn.Sequential(
            nn.Linear(320, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, image: torch.Tensor, age: torch.Tensor, sex: torch.Tensor) -> torch.Tensor:
        # ConvNeXt's original classifier owns the final flatten operation. It
        # is replaced by Identity for this multimodal head, so pool and flatten
        # the feature map explicitly before the projection layer.
        image_features = self.backbone.features(image)
        image_features = self.backbone.avgpool(image_features)
        image_features = torch.flatten(image_features, 1)
        image_embedding = self.image_projection(image_features)
        demographic_embedding = self.demographic_encoder(torch.cat((self.age_encoder(age), self.sex_embedding(sex)), dim=1))
        return self.regressor(torch.cat((image_embedding, demographic_embedding), dim=1)).squeeze(1)

    def freeze_backbone(self, frozen: bool = True) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = not frozen
