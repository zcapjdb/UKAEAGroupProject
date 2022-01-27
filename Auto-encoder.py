import numpy as np 
import pandas as pd 
import torch.nn as nn

class AutoEncoder(nn.Model):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(15,20),
            nn.ReLu(),
            nn.Linear(20, 10),
            nn.ReLu(), 
            nn.Linear(10,5), 
            nn.ReLu(), 
            nn.Linear(5, 3),
        )

        self.decoder = nn.Sequential(
            nn.Linear(3,5),
            nn.ReLu(), 
            nn.Linear(5, 10), 
            nn.ReLu(), 
            nn.Linear(10, 20), 
            nn.ReLu(), 
            nn.Linear(20,15)
        )

        def forward(self, x): 
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

