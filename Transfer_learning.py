import torch
import torch.nn as nn
import numpy as np

# Multi-level domain classifier
class DomainAdversarialClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DomainAdversarialClassifier, self).__init__()
        # Define the first fully connected layer
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Define the second fully connected layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Define the third fully connected layer for binary classification
        self.fc3 = nn.Linear(hidden_dim, 2)
        # Define the ReLU activation function
        self.relu = nn.ReLU()
        # Define the dropout layer with a dropout rate of 0.5
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, alpha):
        # Apply the Gradient Reversal Layer with the given alpha
        x = GradientReversalLayer.apply(x, alpha)
        # Pass through the first fully connected layer with ReLU activation
        x = self.relu(self.fc1(x))
        # Apply dropout
        x = self.dropout(x)
        # Pass through the second fully connected layer with ReLU activation
        x = self.relu(self.fc2(x))
        # Pass through the third fully connected layer to get the final output
        x = self.fc3(x)
        return x 
    
# Gradient Reversal Layer
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        # Save the alpha value for use in the backward pass
        ctx.alpha = alpha
        # Return the input as is (no change in the forward pass)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse the gradient by multiplying with -alpha
        return grad_output.neg() * ctx.alpha, None

# Dynamic adjustment of alpha
def adjust_alpha(epoch, max_alpha=1.0, gamma=10.0):
    # Adjust alpha dynamically based on the epoch using a sigmoid function
    return max_alpha * (2.0 / (1.0 + np.exp(-gamma * epoch)) - 1.)