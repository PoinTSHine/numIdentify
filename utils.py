import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        print(f"using device:{device_name}")
    else:
        device = torch.device("cpu")
        print("using device: cpu")
    return device

def load_mnist_data(batch_size=64, data_path="./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root=data_path, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=data_path, train=False, transform=transform, download=True)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    print("train_dataset: {}\ntest_dataset: {}".format(len(train_dataset), len(test_dataset)))
    
    return train_loader, test_loader

__all__ = ['get_device', 'load_mnist_data']