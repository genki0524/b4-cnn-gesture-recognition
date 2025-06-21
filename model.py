import torch
from torch import nn
from torchvision import *

class SplittedResNet18(nn.Module):
    def __init__(self,model):
        super().__init__()

        self.cnn = nn.Sequential(*list(model.children())[:-1])
        self.flatten = nn.Flatten()
        self.fc = model.fc
    
    def forward(self, x: torch.Tensor):
        representation = self.flatten(self.cnn(x))
        output = self.fc(representation)
        return representation, output





        
