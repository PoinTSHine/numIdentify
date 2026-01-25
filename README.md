# MNIST数字识别

一个基于PyTorch的MNIST手写数字识别应用，包含训练脚本和图形用户界面。

## 功能特点

- 基于卷积神经网络(CNN)的手写数字识别
- 支持在GPU或CPU上运行
- 提供直观的图形用户界面，可直接手绘数字进行识别
- 显示数字识别的概率分布
- 支持清屏和重新绘制功能

## 环境要求

- Python 3.10.16
- PyTorch
- PyQt5
- NumPy
- PIL
- matplotlib

详细依赖见 `requirements.txt` 文件。

## 安装与运行

### 克隆或下载项目

```bash
git clone https://github.com/PoinTSHine/numIdentify.git
cd numIdentify
```

### 安装环境、依赖

```bash
conda create -n numIdentify python=3.10.16
conda activate numIdentify
pip install -r requirements.txt
```

### 运行应用

#### 训练模型

```bash
python train.py
```

结束训练后，会生成 `.\mnist_cnn.pth` 文件。

#### 运行图形界面

```bash
python gui.py
```

运行时会自动加载 `.\mnist_cnn.pth` 模型文件。如果文件不存在，会提示错误并退出。

## 项目结构

```bash
numIdentify/
  ├── gui.py            # 图形用户界面程序
  ├── config.py         # 配置文件
  ├── model.py          # 神经网络模型定义
  ├── train.py          # 模型训练脚本
  ├── utils.py          # 工具函数
  ├── requirements.txt  # 项目依赖
  └── README.md         # 项目说明文档
```

## 使用说明

### config.py 配置参数

- `batch_size`: 批次大小，默认64
- `learning_rate`: 学习率，默认0.001
- `num_epochs`: 训练轮数，默认10
- `num_classes`: 类别数量，固定为10
- `model_save_path`: 模型保存路径，默认"./mnist_cnn.pth"
