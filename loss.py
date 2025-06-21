from torch import nn
import torch
from torch import linalg as LA
import torch.nn.functional as F

class CrossEntropyOSLoss(nn.Module):
    def __init__(self, regularization_param):
        super().__init__()
        self.Cross_Entropy_Loss = nn.CrossEntropyLoss()
        self.alpha = regularization_param

    def forward(self, output, representation, target, device):

        # 通常のクロスエントロピー誤差
        CEL_value = self.Cross_Entropy_Loss(output, target)

        # Orthogonal Sphere Regularization
        normalized_representation = F.normalize(representation, p = 2, dim = 1)
        OS_value = torch.add(
            torch.matmul(torch.t(normalized_representation), normalized_representation),
            torch.eye(normalized_representation.size()[1], device = device),
            alpha = -1
        )
        OS_value = self.alpha*LA.norm(OS_value, ord = "fro")

        return CEL_value + OS_value