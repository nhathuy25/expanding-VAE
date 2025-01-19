import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn
from model import VAE, VAE_expanding
import matplotlib.pyplot as plt

# configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 784
HIDDEN_DIM = 256
LATENT_DIM = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
BATCH_SIZE = 32

data_transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST(root='dataset/', train=True, transform=data_transform, download=True)

train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
model = VAE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM).to(DEVICE)

model.load_state_dict(torch.load('./saved_model/VAE_2hid_20eps.pth', map_location=torch.device('cpu'), weights_only=True), strict=False)

model.eval()

def get_samples_for_digits(dataset, num_digits=10):
    """
    Get one image for each digit from 0 to 9 in the MNIST dataset.
    
    Args:
        dataset: MNIST dataset.
        num_digits: Number of digits to retrieve (default is 10: from 0 to 9).
    
    Returns:
        images: List of images corresponding to each digit.
        labels: List of labels for each digit.
    """
    images = []
    labels = []
    digit_set = set()
    
    for img, label in dataset:
        if label.item() not in digit_set:
            images.append(img.view(28, 28).numpy())  # Save the image
            labels.append(label.item())  # Save the label
            digit_set.add(label.item())  # Add digit to the set of seen digits
            
        if len(digit_set) == num_digits:  # Stop when all 10 digits are collected
            break
    
    # Sort the images by their labels
    sorted_images_and_labels = sorted(zip(labels, images), key=lambda x: x[0])
    sorted_labels, sorted_images = zip(*sorted_images_and_labels)
    
    return list(sorted_images), list(sorted_labels)

def get_latent_vector_from_image(model, image, device=torch.device("cpu")):
    """
    Compute the latent vector z from an input image using the VAE model.

    Args:
        model: The trained VAE model.
        image: The input image, as a torch.Tensor with shape (1, 28, 28) or (28, 28).
        device: The device to run on (CPU or GPU).

    Returns:
        z: The latent vector z with noise epsilon.
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Reshape the image if necessary
        if len(image.shape) == 2:  # If the image is (28, 28)
            image = image.unsqueeze(0)  # Add batch dimension
        if len(image.shape) == 3:  # If the image is (1, 28, 28)
            image = image.unsqueeze(0)  # Add batch dimension for batch=1

        image = image.to(device)  # Move image to device
        image = image.view(image.size(0), -1)  # Flatten the image into (batch_size, INPUT_DIM)

        # Compute mu and log_sigma through the encoder
        mu, log_sigma = model.encoder(image)
        sigma = torch.exp(0.5 * log_sigma)  # Convert from log_sigma to sigma

        # Create epsilon from a normal distribution
        epsilon = torch.randn_like(sigma).to(device)

        # Reparameterization to compute z
        z = mu + epsilon * sigma

    return z

def generate_image_from_latent(model, latent_dim, device=torch.device("cpu")):
    """
    Generate a new image from a random latent vector using the VAE decoder.

    Args:
        model: The trained VAE model.
        latent_dim: The latent space dimensionality (LATENT_DIM).
        device: The device to run on (CPU or GPU).

    Returns:
        generated_image: The generated image (numpy array, shape (28, 28)).
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Create a random latent vector z from a normal distribution
        z = torch.randn(1, latent_dim).to(device)  # Shape (1, LATENT_DIM)
        
        # Use the decoder to generate a new image
        generated_image = model.decoder(z)  # Reconstruct from z
        generated_image = generated_image.view(28, 28).cpu().numpy()  # Reshape and convert to numpy

    return generated_image

def generate_multiple_images(model, latent_dim, num_samples, device=torch.device("cpu")):
    """
    Generate multiple new images from random latent vectors.

    Args:
        model: The trained VAE model.
        latent_dim: The latent space dimensionality (LATENT_DIM).
        num_samples: The number of images to generate.
        device: The device to run on (CPU or GPU).

    Returns:
        images: A list of generated images (numpy arrays, each with shape (28, 28)).
    """
    model.eval()
    images = []
    with torch.no_grad():
        # Create num_samples random latent vectors z
        z = torch.randn(num_samples, latent_dim).to(device)  # Shape (num_samples, LATENT_DIM)
        
        # Use the decoder to generate images
        generated_data = model.decoder(z)  # Reconstruct from z
        generated_data = generated_data.view(-1, 28, 28).cpu().numpy()  # Reshape and convert to numpy
        
        images = [img for img in generated_data]

    return images

# Retrieve samples from the MNIST dataset
mnist_loader = DataLoader(dataset, batch_size=1, shuffle=False)  # No shuffling
images, labels = get_samples_for_digits(mnist_loader)

# Generate 10 new images
generated_images = generate_multiple_images(model, LATENT_DIM, num_samples=10, device=DEVICE)

# Display both original (MNIST) and generated images
fig, axes = plt.subplots(2, 10, figsize=(15, 6))  # 2 rows, 10 columns

# Display the original MNIST images
for i in range(10):
    axes[0, i].imshow(images[i], cmap='gray')  # Original MNIST image
    axes[0, i].set_title(f"Digit: {labels[i]}")  # Label of the image
    axes[0, i].axis('off')

# Display the generated images
for i in range(10):
    axes[1, i].imshow(generated_images[i], cmap='gray')  # Generated image
    axes[1, i].set_title(f"Generated {i+1}")  # Title (Generated)
    axes[1, i].axis('off')

plt.tight_layout()  # Adjust layout
plt.show()
