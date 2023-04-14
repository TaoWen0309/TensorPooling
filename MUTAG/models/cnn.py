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
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    def forward(self, PI):
        feature = self.features(PI)
        return feature

# compute output dim given the above kernel_size and stride
def cnn_output_dim(dim_in):
    tmp_dim = int((dim_in-2)/2)+1
    output_dim = int((tmp_dim-2)/2)+1
    return output_dim