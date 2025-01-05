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
import copy
from typing import Dict, List

''' Configurations'''
INPUT_DIM = 784
HIDDEN_DIM_1 = 256
HIDDEN_DIM_2 = 128
LATENT_DIM = 32
GROW_EPOCH = 11

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
BATCH_SIZE = 32

''' Load the MNIST dataset '''
data_transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST(root='dataset/', train=True, transform=data_transform, download=True)

train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

''' Define Loss functions for VAE 
    - Reconstruction difference, we use the Binary Cross Entropy (BCE) loss
    - Kull
'''
# BCE Loss
BCE_loss = nn.BCELoss(reduction = 'sum')

# Loss function
def loss_function(x, x_recon, mean, log_var):

    reproduction_loss = BCE_loss(x_recon, x)
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + KLD

''' Trainning per epoch function
'''
def train(model, tqdm_loop, loss_function, optimizer):
    train_loss = 0
    for batch_idx, (x, _) in tqdm_loop:
        # flatten the imput image
        x = x.view(-1, 28*28)
        x = x.to(DEVICE)
        
        x_recon, mu, log_var = model(x)

        # compute loss functions    
        total_loss = loss_function(x, x_recon, mu, log_var)

        # backprop
        train_loss += total_loss.item()

        #print(f"Total loss before .item(): {total_loss}")

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        tqdm_loop.set_postfix(loss=train_loss/(batch_idx*BATCH_SIZE+1))
    
    return train_loss/(batch_idx*BATCH_SIZE+1)

''' 
Evaluation metric
Inputs: - model: the model to be evaluated
        - testLoader: the data loader of test dataset
        - 
'''
def measure_variance():
    pass

''' 
Adding nodes to the expands layers function
Inputs: - model: the model to be expanded
        - idx_layer: the index of the layer to be expanded
        - nb_node: the number of nodes to be added
        - epoch: the current epoch
        - bool_encoder: True if the layer is in the encoder
                        False if the layer is in the decoder
'''
def func_expand_layer(model, idx_layer, nb_node, epoch, bool_encoder = True):
    model.expand_layer(idx_layer, nb_node, bool_encoder)
    if bool_encoder:
        print(f'Encoder layer {idx_layer} expanded with {nb_node} nodes at epoch {epoch}')
    else:
        print(f'Decoder layer {idx_layer} expanded with {nb_node} nodes at epoch {epoch}')

''' Define the model '''
encoder_config = [HIDDEN_DIM_1, HIDDEN_DIM_2, LATENT_DIM]
decoder_config = [HIDDEN_DIM_2, HIDDEN_DIM_1, INPUT_DIM]

# Define the date
date = time.strftime("%Y%m%d")

# Set the name for the model for saving
model_name = f'VAE_2hid_{NUM_EPOCHS}eps_{date}'

model = VAE_expanding((28, 28), device=DEVICE)
model.construct(encoder_config, decoder_config, False)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

''' 
TRAINING PART 
'''
# Configurations
GROW_EPOCH = 5
NB_NODE_ADD_1 = 32
NB_NODE_ADD_2 = 16

# Mark start time
start_time = time.time()

train_info = {
    'growth_epoch': [],
    'loss_list': [],
    'accuracy': []
}

train_loss_list = []
train_loss = 0

model.train()
for epoch in range(NUM_EPOCHS):

    if (epoch+1) % GROW_EPOCH == 0 and epoch != 0:
        # Adding growth epoch info
        train_info['growth_epoch'].append(epoch)
        model_growth = copy.deepcopy(model) # Explain
        # Adding nodes to the first layer of encoder
        func_expand_layer(model_growth, 0, NB_NODE_ADD_1, epoch, True)
        # Adding nodse to the second layer of encoder
        func_expand_layer(model_growth, 2, NB_NODE_ADD_2, epoch, True)
        growth_optimizer = torch.optim.Adam(model_growth.parameters(), lr=LEARNING_RATE)

    loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{NUM_EPOCHS}', leave=False)
    train_loss = train(model, loop, loss_function, optimizer)
    train_info['loss_list'].append(train_loss/len(train_loader))

''' End of training & saving the model '''

# Mark end time
end_time = time.time()
elapsed_time = end_time - start_time

print(f"Training completed in {elapsed_time:.2f} seconds")

# Save the training loss list to a file
with open(f'saved_loss/train_loss_{model_name}.json', 'w') as f:
    json.dump(train_info['loss_list'], f)

# save the model
torch.save(model.state_dict(), f'saved_model/{model_name}.pth')