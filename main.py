"""
This script is used to train the VAE model on the MNIST dataset. Used on the last internship at LIFAT.
Huy
"""

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn
from model import VAE_expanding
from tqdm import tqdm
import time
import json
import copy
from typing import Dict, List
import numpy as np

''' MODEL '''

# Configurations
INPUT_DIM = 784
HIDDEN_DIM_1 = 64
HIDDEN_DIM_2 = 32
LATENT_DIM = 20
GROW_EPOCH = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3
BATCH_SIZE = 128


# Expanding configurations
L_SAMPLE = 1
NB_NODE_ADD_1 = 64
NB_NODE_ADD_2 = 32

# Set the name for the model for saving
model_name = f'VAE_FashionMNIST_{NUM_EPOCHS}_{BATCH_SIZE}_{LATENT_DIM}_{L_SAMPLE}'

# ------------------------

''' Load the MNIST dataset '''
data_transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST(root='dataset/', train=True, transform=data_transform, download=True)

train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

''' Define Loss functions for VAE 
    - Reconstruction difference, we use the Binary Cross Entropy Loss function BCE(x, decoder(z))
    - Kullback - Leibler divergence KL(q(z|x) || p(z)) between the posterior q(z|x) and the prior p(z)
'''
# BCE Loss
BCE_loss = nn.BCELoss(reduction = 'sum')

# Loss function
def loss_function(x, x_recon, mean, log_var):
    reproduction_loss = BCE_loss(x_recon, x)
    KLD      =  - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + KLD

''' Trainning per epoch function
'''
def train(model, tqdm_loop, loss_function, optimizer):
    train_loss = 0
    sum_elbo = 0
    elbo = 0
    for batch_idx, (x, _) in tqdm_loop:
        # flatten the imput image
        x = x.view(-1, 28*28)
        x = x.to(DEVICE)
        # pass the input to the model
        x_recon, mu, log_var = model(x)

        # calculate the loss
        total_loss = loss_function(x, x_recon, mu, log_var)

        '''reproduction_loss = BCE_loss(x_recon, x)
        KLD =  - 0.5 * torch.sum(1+ log_var - mu.pow(2) - log_var.exp())
        total_loss = reproduction_loss + KLD'''

        # sum the losses over the batches
        train_loss += total_loss.item()

        # calculate elbo for each batch
        elbo = measure_ELBO(x, model, L_SAMPLE)
        sum_elbo += elbo

        #print(f"Total loss before .item(): {total_loss}")

        optimizer.zero_grad()   # clear the gradients
        total_loss.backward()   # back-propagation
        optimizer.step()        # update the weights

        # update the progress bar
        #tqdm_loop.set_postfix(loss=train_loss/(batch_idx*BATCH_SIZE+1)) primary code
        tqdm_loop.set_postfix(
            loss=train_loss / (batch_idx * BATCH_SIZE + 1),
            gradient=torch.norm(torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None]))
        )


    # Return: - the average loss over number of data points
    #         - the average elbo over number of batches
    return train_loss/(len(tqdm_loop)*BATCH_SIZE), elbo/(BATCH_SIZE)

'''
Measuring Evidence Lower Bound (ELBO) for each datapoint x_i with L samples from the latent space
Inputs: - x: the datapoint
        - model: the model to be evaluated
        - L: the number of samples to be drawn from the latent space

Outputs: - scalar value (unbiased stochastic estimate lower bound on logp_theta(x))
'''
def measure_ELBO(x, model, L):
    
    _, mu_z, logvar_z = model(x)
    ELBO = 0
    for l in range(L):
        
        z = model.reparametrization(mu_z, logvar_z)

        ELBO += -loss_function(x, model.decode(z), mu_z, logvar_z)

    ELBO = ELBO/L
    return ELBO.cpu().detach().numpy()    


''' 
Adding nodes to the expands an ENCODER'S LAYER function
Inputs: - model: the model to be expanded
        - idx_layer: the index of the layer to be expanded
        - nb_node: the number of nodes to be added
        - epoch: the current epoch
        - bool_encoder: True if the layer is in the encoder
                        False if the layer is in the decoder
'''
def func_expand_layer(model, idx_layer, nb_node, epoch):
    model.expand_layer(idx_layer, nb_node)
    print(f'Encoder layer {idx_layer} expanded with {nb_node} nodes at epoch {epoch}')

''' Define the model '''
encoder_config = [HIDDEN_DIM_1, HIDDEN_DIM_2, LATENT_DIM]
decoder_config = [128, 256, INPUT_DIM]

# Define the date
date = time.strftime("%Y%m%d")

model = VAE_expanding((28, 28), device=DEVICE)
model.construct(encoder_config, decoder_config, False)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

for (x, y) in train_loader:
    print('Min Pixel Value: {} \nMax Pixel Value: {}'.format(x.min(), x.max()))
    print('Mean Pixel Value {} \nPixel Values Std: {}'.format(x.float().mean(), x.float().std()))
    break
''' 
TRAINING PART 
'''

# ------------------------
# Mark start time
start_time = time.time()

train_info = {
    'growth_epoch': [],
    'loss_list': [],
    'loss_growth_list': [], # Adding loss list for the growth model
    'elbos': [],
    'elbos_growth': [] # Adding elbo list for the growth model
}

train_loss_list = []
train_loss = 0
train_loss_growth_list = []
train_loss_growth = 0
train_elbos = []
train_elbos_growth = []

model.train()
for epoch in range(NUM_EPOCHS):
    """ Expanding the model if:
        - The current epoch is a multiple of GROW_EPOCH
        - The current epoch is not the first epoch
        - The current epoch is not the last epoch 
    """
    if (epoch+1) % GROW_EPOCH == 0 and epoch != 0 and epoch+1 != NUM_EPOCHS:
        # Adding growth epoch info
        train_info['growth_epoch'].append(epoch)
        model_growth = copy.deepcopy(model) if epoch+1 == GROW_EPOCH else copy.deepcopy(model_growth)
        # Adding nodes to the first layer of encoder
        func_expand_layer(model_growth, 0, NB_NODE_ADD_1, epoch+1)
        # Adding nodse to the second layer of encoder
        func_expand_layer(model_growth, 2, NB_NODE_ADD_2, epoch+1)
        growth_optimizer = torch.optim.Adam(model_growth.parameters(), lr=LEARNING_RATE)
        #growth_optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    # Training
    loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{NUM_EPOCHS} (model)', leave=False)
    train_loss, elbos = train(model, loop, loss_function, optimizer)
    train_loss_list.append(train_loss)
    train_elbos.append(elbos)

    if 'model_growth' in locals():
        # Train the growth model parallelly
        loop_growth = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{NUM_EPOCHS} (model_growth)', leave=False)
        train_loss_growth, elbos_growth = train(model_growth, loop_growth, loss_function, growth_optimizer)
        train_loss_growth_list.append(train_loss_growth)
        train_elbos_growth.append(elbos_growth)

# Copy the initial losses (before first growing epoch) to the growth model's loss list and elbos list
initial_losses = train_loss_list[:GROW_EPOCH-1]
train_loss_growth_list = initial_losses + train_loss_growth_list

initial_elbos = train_elbos[:GROW_EPOCH-1]
train_elbos_growth = initial_elbos + train_elbos_growth    

# Save the losses and elbos to the dictionary
train_info['loss_list'] = train_loss_list
train_info['elbos'] = train_elbos
if 'model_growth' in locals():
    train_info['loss_growth_list'] = train_loss_growth_list
    train_info['elbos_growth'] = train_elbos_growth

''' End of training & saving the model '''
print(f"Final model: ", model)
print(f"Final model growth: ", model_growth)
print(f'Losses list size: {len(train_info["loss_list"])}')
print(f'Losses growth list size: {len(train_info["loss_growth_list"])}')


# Mark end time
end_time = time.time()
elapsed_time = end_time - start_time

print(f"Training completed in {elapsed_time:.2f} seconds")

# Save the training loss list to a file
with open(f'saved_train-info/train_info{model_name}.json', 'w') as f:
    json.dump(train_info, f)

# save the model
torch.save(model_growth.state_dict(), f'saved_model/{model_name}.pth')

#------------------------ Plot ------------------------
import matplotlib.pyplot as plt
# Plot the training loss
fig, ax1 = plt.subplots(figsize=(8, 5))
epochs = range(1, len(train_info['loss_list']) + 1)

# Plot LOSS on the first y-axis
line1, = ax1.plot(epochs, train_info['loss_list'], label='Training Loss', color='b')
line2, = ax1.plot(epochs, train_info['loss_growth_list'], label='Training Loss Growth', color='g')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.tick_params(axis='y')

# Add vertical dashed lines at x % 5 == 0
for x in epochs:
    if x % GROW_EPOCH == 0 and x != NUM_EPOCHS:
        ax1.axvline(x=x, color='gray', linestyle='--', linewidth=0.5)

# Set x-axis to display every 5th epoch
ax1.set_xticks(np.arange(0, len(epochs) + 1, 5))

# Create a second y-axis for ELBO
ax2 = ax1.twinx()
line3, = ax2.plot(epochs, train_info['elbos'], label='ELBO', color='r')
line4, = ax2.plot(epochs, train_info['elbos_growth'], label='ELBO Growth', color='y')
ax2.set_ylabel('ELBO')
ax2.tick_params(axis='y')

# Combine legends
lines = [line1, line2, line3, line4]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='upper left', prop={'size': 8})

# Add labels and title
plt.title('Training Loss and ELBO by Epoch')
fig.tight_layout()  # Adjust layout to make room for both y-axes
plt.savefig(f'img/{model_name}.png')
plt.show()