"""
This script is used to train the VAE model on the MNIST dataset. Used on the last internship at LIFAT.
Huy
"""

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn
from model import VAE, VAE_expanding
from tqdm import tqdm
import time
import json

# configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 784
HIDDEN_DIM = 256
LATENT_DIM = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
BATCH_SIZE = 32

data_transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST(root='dataset/', train=True, transform=data_transform, download=True)

train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
model = VAE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM).to(DEVICE)

encoder_config = [256,  32]
decoder_config = [256, 784]

# Set the name for the model for saving
model_name = f'VAE_1hid_{NUM_EPOCHS}eps'

model = VAE_expanding((28, 28), device=DEVICE)
model.construct(encoder_config, decoder_config, False)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

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

BCE_loss = nn.BCELoss(reduction = 'sum')

def loss_function(x, x_recon, mean, log_var):

    reproduction_loss = BCE_loss(x_recon, x)
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + KLD

# Trainning

# Trainning
start_time = time.time()

train_loss_list = []
model.train()
for epoch in range(NUM_EPOCHS):
    loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{NUM_EPOCHS}', leave=False)
    train_loss = 0
    for batch_idx, (x, _) in loop:
        # flatten the imput image
        x = x.view(-1, 28*28)
        x = x.to(DEVICE)
        
        x_recon, mu, log_var = model(x)

        # compute loss functions    
        #total_loss = loss_function(x, x_recon, mu, log_var)
        
        reproduction_loss = BCE_loss(x_recon, x)
        KL_div = - 0.5 * torch.sum(1+ log_var - mu.pow(2) - log_var.exp())
        total_loss = reproduction_loss + KL_div

        # backprop
        train_loss += total_loss.item()

        #print(f"Total loss before .item(): {total_loss}")

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        loop.set_postfix(loss=train_loss/(batch_idx*BATCH_SIZE+1))
    
    train_loss_list.append(train_loss / (batch_idx*BATCH_SIZE+1))

# Mark end time
end_time = time.time()
elapsed_time = end_time - start_time

print(f"Training completed in {elapsed_time:.2f} seconds")

# Save the training loss list to a file
with open(f'saved_loss/train_loss_{model_name}.json', 'w') as f:
    json.dump(train_loss_list, f)

# save the model
torch.save(model.state_dict(), f'saved_model/{model_name}.pth')