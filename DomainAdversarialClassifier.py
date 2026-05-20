import torch
import torch.nn as nn
import numpy as np

class DomainAdversarialClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DomainAdversarialClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, alpha):
        x = GradientReversalLayer.apply(x, alpha)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x 
    
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def adjust_alpha(epoch, max_alpha=1.0, gamma=10.0):
    return 2.0 / (1.0 + np.exp(-gamma * epoch)) - 1.0