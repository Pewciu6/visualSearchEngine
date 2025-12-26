import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class EmbeddingNet(nn.Module):
    def __init__(self, embedding_size: int = 128, pretrained: bool = True):
        super(EmbeddingNet, self).__init__()
        
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        fc_in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(fc_in_features, embedding_size)
        
    def forward(self, x):
        x = self.backbone(x)
        x = F.normalize(x, p=2, dim=1)
        
        return x