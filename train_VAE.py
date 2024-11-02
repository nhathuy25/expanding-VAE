"""
This script is used to train the VAE model on the MNIST dataset. Used on the last internship at LIFAT.
Huy
"""

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn
from model import VAE
from tqdm import tqdm
import math

# configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 784
HIDDEN_DIM = 256
LATENT_DIM = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
BATCH_SIZE = 64

data_transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST(root='dataset/', train=True, transform=data_transform, download=True)

train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
model = VAE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# loss function
loss_fn = nn.BCELoss(reduction='sum')

#Verify pixels in data loader:
train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
)
torch.manual_seed(5576)
for (x, y) in train_loader:
    print('Min Pixel Value: {} \nMax Pixel Value: {}'.format(x.min(), x.max()))
    print('Mean Pixel Value {} \nPixel Values Std: {}'.format(x.float().mean(), x.float().std()))
    break

# Trainning
model.train()
for epoch in range(NUM_EPOCHS):
    loop = tqdm(enumerate(train_loader))
    train_loss = 0
    for batch_idx, (x, _) in loop:
        # flatten the imput image
        x = x.view(-1, 28*28)
        x = x.to(DEVICE)
        x_recon, mu, log_var = model(x)

        # compute loss functions    
        reproduction_loss = loss_fn(x_recon, x)
        KL_div = - 0.5 * torch.sum(1+ log_var - mu.pow(2) - log_var.exp())

        # backprop
        total_loss = reproduction_loss + KL_div
        train_loss += total_loss.item()

        #print(f"Total loss before .item(): {total_loss}")

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        loop.set_postfix(loss=train_loss/(batch_idx*BATCH_SIZE+1))

torch.save(model.state_dict(), 'saved_model/VAE_2hid_15eps.pth')