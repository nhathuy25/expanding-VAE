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
    Lấy mỗi chữ số từ 0 đến 9 trong tập MNIST.
    
    Args:
        dataset: Tập dữ liệu MNIST.
        num_digits: Số lượng chữ số cần lấy (mặc định 10: từ 0 đến 9).
    
    Returns:
        images: Danh sách hình ảnh tương ứng với từng chữ số.
        labels: Danh sách nhãn của từng chữ số.
    """
    images = []
    labels = []
    digit_set = set()
    
    for img, label in dataset:
        if label.item() not in digit_set:
            images.append(img.view(28, 28).numpy())  # Lưu ảnh
            labels.append(label.item())  # Lưu nhãn
            digit_set.add(label.item())  # Thêm chữ số vào tập đã thấy
            
        if len(digit_set) == num_digits:  # Khi đủ 10 chữ số thì dừng
            break
    
    # Sắp xếp hình ảnh theo thứ tự tăng dần của nhãn
    sorted_images_and_labels = sorted(zip(labels, images), key=lambda x: x[0])
    sorted_labels, sorted_images = zip(*sorted_images_and_labels)
    
    return list(sorted_images), list(sorted_labels)

def get_latent_vector_from_image(model, image, device=torch.device("cpu")):
    """
    Tính toán vector z từ một hình ảnh đầu vào sử dụng mô hình VAE.

    Args:
        model: Mô hình VAE đã được huấn luyện.
        image: Ảnh đầu vào, dạng torch.Tensor với shape (1, 28, 28) hoặc (28, 28).
        device: Thiết bị thực thi (CPU hoặc GPU).

    Returns:
        z: Vector latent z với noise epsilon.
    """
    model.eval()  # Đặt mô hình ở chế độ đánh giá
    with torch.no_grad():
        # Đưa ảnh về đúng định dạng
        if len(image.shape) == 2:  # Nếu ảnh là (28, 28)
            image = image.unsqueeze(0)  # Thêm batch dimension
        if len(image.shape) == 3:  # Nếu ảnh là (1, 28, 28)
            image = image.unsqueeze(0)  # Thêm batch dimension cho batch=1

        image = image.to(device)  # Chuyển ảnh sang thiết bị
        image = image.view(image.size(0), -1)  # Flatten ảnh thành (batch_size, INPUT_DIM)

        # Tính toán mu và log_sigma thông qua encoder
        mu, log_sigma = model.encoder(image)
        sigma = torch.exp(0.5 * log_sigma)  # Chuyển từ log_sigma thành sigma

        # Tạo epsilon từ phân phối chuẩn
        epsilon = torch.randn_like(sigma).to(device)

        # Tái tham số hóa để tính z
        z = mu + epsilon * sigma

    return z

def generate_image_from_latent(model, latent_dim, device=torch.device("cpu")):
    """
    Sinh một hình ảnh mới từ latent vector ngẫu nhiên sử dụng decoder của VAE.

    Args:
        model: Mô hình VAE đã được huấn luyện.
        latent_dim: Kích thước không gian latent (LATENT_DIM).
        device: Thiết bị thực thi (CPU hoặc GPU).

    Returns:
        generated_image: Hình ảnh được tạo ra (dạng numpy array, shape (28, 28)).
    """
    model.eval()  # Đặt mô hình ở chế độ đánh giá
    with torch.no_grad():
        # Tạo vector z ngẫu nhiên từ phân phối chuẩn
        z = torch.randn(1, latent_dim).to(device)  # Kích thước (1, LATENT_DIM)
        
        # Dùng decoder để sinh ảnh mới
        generated_image = model.decoder(z)  # Tái tạo từ z
        generated_image = generated_image.view(28, 28).cpu().numpy()  # Reshape và chuyển sang numpy

    return generated_image

def generate_multiple_images(model, latent_dim, num_samples, device=torch.device("cpu")):
    """
    Sinh nhiều hình ảnh mới từ các latent vector ngẫu nhiên.

    Args:
        model: Mô hình VAE đã được huấn luyện.
        latent_dim: Kích thước không gian latent (LATENT_DIM).
        num_samples: Số lượng ảnh cần tạo.
        device: Thiết bị thực thi (CPU hoặc GPU).

    Returns:
        images: Danh sách các ảnh được tạo (numpy arrays, mỗi ảnh có shape (28, 28)).
    """
    model.eval()
    images = []
    with torch.no_grad():
        # Tạo num_samples vector z ngẫu nhiên
        z = torch.randn(num_samples, latent_dim).to(device)  # Kích thước (num_samples, LATENT_DIM)
        
        # Dùng decoder để sinh ảnh
        generated_data = model.decoder(z)  # Tái tạo từ z
        generated_data = generated_data.view(-1, 28, 28).cpu().numpy()  # Reshape và chuyển sang numpy
        
        images = [img for img in generated_data]

    return images

# Lấy mẫu từ tập MNIST
mnist_loader = DataLoader(dataset, batch_size=1, shuffle=False)  # Không xáo trộn
images, labels = get_samples_for_digits(mnist_loader)

# Sinh 10 ảnh mới
generated_images = generate_multiple_images(model, LATENT_DIM, num_samples=10, device=DEVICE)

# Hiển thị cả ảnh gốc (MNIST) và ảnh được sinh
fig, axes = plt.subplots(2, 10, figsize=(15, 6))  # 2 hàng, 10 cột

# Hiển thị các ảnh gốc từ MNIST
for i in range(10):
    axes[0, i].imshow(images[i], cmap='gray')  # Ảnh gốc MNIST
    axes[0, i].set_title(f"Digit: {labels[i]}")  # Nhãn của ảnh
    axes[0, i].axis('off')

# Hiển thị các ảnh được sinh từ latent space
for i in range(10):
    axes[1, i].imshow(generated_images[i], cmap='gray')  # Ảnh được sinh
    axes[1, i].set_title(f"Generated {i+1}")  # Tiêu đề (Generated)
    axes[1, i].axis('off')

plt.tight_layout()  # Điều chỉnh layout
plt.show()