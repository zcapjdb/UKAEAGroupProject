from turtle import forward
import pandas as pd 
import numpy as np 
import torch.nn as nn
import torch

class QLKNN(nn.Module):
    def __init__(self, n_input):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_input,128), 
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )
    
    def forward(self, x):
        X = self.model(x)

        return X

def foward_step(X, y, loss_fn, optimizer, model):
    pred = model.forward(X)
    loss = loss_fn(pred, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def loss_function(y, y_hat):
    # Loss function missing regularization term (to be added using Adam optimizer)
    lambda_stab = 1e-3
    k_stab = 5
    if y.sum() == 0: 
        c_good = 0
        c_stab = torch.mean(y_hat - k_stab)

    else: 
        c_good = torch.mean(torch.square(y - y_hat))
        c_stab = 0
    return c_good + lambda_stab*k_stab


