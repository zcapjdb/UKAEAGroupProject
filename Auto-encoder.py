import numpy as np 
import pandas as pd 
import torch.nn as nn

class Encoder(nn.Model): 
    def __init__(self, latent_dims):
        super().__init__()
        self.latent_dims = latent_dims

        self.encoder = nn.Sequential(
            nn.Linear(15,20),
            nn.ReLu(),
            nn.Linear(20, 10),
            nn.ReLu(), 
            nn.Linear(10,5), 
            nn.ReLu(),
            nn.Linear(5, self.latent_dims),
        )
    
    def forward(self, x): 
        encoded = self.encoder(x)

        return encoded

class Decoder(nn.Model):
        def __init__(self, latent_dims):
            super().__init__()
            self.latent_dims = latent_dims

            self.decoder = nn.Sequential(
            nn.Linear(latent_dims,5),
            nn.ReLu(), 
            nn.Linear(5, 10), 
            nn.ReLu(), 
            nn.Linear(10, 20), 
            nn.ReLu(), 
            nn.Linear(20,15)
        )

        def forward(self, encoded):
            decoded = self.decoder(encoded)

            return decoded



    

class AutoEncoder(nn.Model):

    def __init__(self, latent_dims, encoder, decoder):
        super().__init__()
        self.latent_dims = latent_dims
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded

