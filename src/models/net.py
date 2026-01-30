import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class EmbeddingNet(nn.Module):
    def __init__(self, architecture : str = 'resnet', embedding_size: int = 128, pretrained: bool = True):
        super().__init__()
        self.architecture = architecture

        if self.architecture == 'resnet':
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            fc_in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(fc_in_features, embedding_size)
        elif architecture == "vit":
            weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
            self.backbone = models.vit_b_16(weights=weights)
            in_features = self.backbone.heads.head.in_features
            self.backbone.heads = nn.Sequential(
                nn.Linear(in_features, embedding_size)
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def forward(self, x):
        x = self.backbone(x)
        x = F.normalize(x, p=2, dim=1)

        return x
