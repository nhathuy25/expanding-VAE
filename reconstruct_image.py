import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import VAE
import matplotlib.pyplot as plt

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 784
HIDDEN_DIM = 256
LATENT_DIM = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
BATCH_SIZE = 32

# Data config
data_transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST(root='dataset/', train=True, transform=data_transform, download=True)

train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
model = VAE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM).to(DEVICE)

# Path to trained model
model_path = './saved_model/VAE_2hid_20eps.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True), strict=False)
model.eval()

# Number of images per digit
IMAGES_PER_DIGIT = 10

# Generate images for each digit from 0 to 9
fig, axes = plt.subplots(10, IMAGES_PER_DIGIT, figsize=(15, 15))
plt.suptitle("Reconstructed Images for Digits 0-9", fontsize=16)
for digit in range(10):
    digit_images = []
    
    # Iterate through the dataset and collect images with the label 'digit'
    for image, label in dataset:
        if label == digit:
            digit_images.append(image)
    
    for i in range(IMAGES_PER_DIGIT):  # Generate 10 images for each digit
        # Take one image from the collection for this digit
        image = digit_images[i].unsqueeze(0).to(DEVICE)
        
        # Generate an image from the input image
        with torch.no_grad():
            # Encode the image into latent space using the encoder
            mean, log_var = model.encoder(image.view(1, -1))  # Encoder returns mean and log-variance
            
            # Sample latent vector from the normal distribution (mean, log_var)
            std = torch.exp(0.5 * log_var) 
            epsilon = torch.randn(1, LATENT_DIM).to(DEVICE) 
            latent_vector = mean + epsilon * std  
            
            # Use the decoder to reconstruct an image from the latent vector
            generated_image = model.decoder(latent_vector)

        # Display the generated image
        ax = axes[digit, i]
        ax.imshow(generated_image.cpu().numpy().reshape(28, 28), cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
plt.show()
