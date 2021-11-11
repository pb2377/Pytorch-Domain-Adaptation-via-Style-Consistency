import torch
import torch.nn as nn


class FeatureConsistency(nn.Module):
    def __init__(self, cosine=False):
        super(FeatureConsistency, self).__init__()
        self.cosine = cosine
        if cosine:
            self.criterion = nn.CosineEmbeddingLoss(reduction='none')
        else:
            self.criterion = torch.nn.MSELoss(reduction='none')
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

    def forward(self, x1, x2):
        if self.cosine:
            return self.criterion(x1, x2, target=torch.ones(1)).mean()
        else:
            return self.criterion(x1, x2).mean()
