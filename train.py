import torch
import torch.nn as nn
import torch.optim as optim

from model import Minst_cnn
from utils import load_mnist_data, get_device

Config = {
    "batch_size": 64,
    "learning_rate": 0.001,
    "num_epochs": 10,
    "num_classes": 10,
    "model_save_path": "./mnist_cnn.pth"
}

def train(model, train_loader, optimizer, criterion, device, num_epoch):
    model.train()
    total_steps = len(train_loader)
    print("\n======== 开始训练 =======")
    for epoch in range(num_epoch):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}"
                    .format(epoch+1, num_epoch, i+1, total_steps, loss.item()))
                
    print("======== 训练结束 =======\n")

def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print("======== 测试结束 =======")
    print("测试集总样本数: {}\n正确识别{}".format(total, correct))
    print("测试集准确率: {:.2f}%".format(acc))

if __name__ == "__main__":
    device = get_device()
    train_loader, test_loader = load_mnist_data(batch_size=Config["batch_size"])
    model = Minst_cnn(num_classes=Config["num_classes"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config["learning_rate"])
    train(model, train_loader, optimizer, criterion, device, Config["num_epochs"])
    test(model, test_loader, device)
    torch.save(model.state_dict(), Config["model_save_path"])
    print("模型已保存至{}".format(Config["model_save_path"]))