import torch
import torch.nn.functional as F
import torch.nn as nn

# apply to PI
class CNN(nn.Module):
    def __init__(self, dim_out):
        super(CNN, self).__init__()
        self.dim_out = dim_out
        self.features = nn.Sequential(
            nn.Conv2d(5, dim_out, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, PI):
        feature = self.features(PI)
        # feature = feature.view(-1, self.dim_out) #B, dim_out
        return feature