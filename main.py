from model import *
import numpy as np
import torch
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder_config = [256,  32]
decoder_config = [256, 784]
model = VAE_expanding((28, 28), device=DEVICE)
model.construct(encoder_config, decoder_config, False)

print(model)

x = torch.randn(64, 784)
'''
x_recon, mu, log_var = model(x.to(DEVICE))
print(x.shape)
print(x_recon.shape)
print(mu.shape)
print(log_var.shape)
'''