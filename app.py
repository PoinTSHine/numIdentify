from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from model import Minst_cnn
from utils import get_device
import base64
from io import BytesIO

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# 加载模型
device = get_device()
model = Minst_cnn(num_classes=10).to(device)
try:
    model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device, weights_only=True))
    print("[note] Model loaded successfully.")
except FileNotFoundError:
    print("[error] Model file 'mnist_cnn.pth' not found.")
    print("[error] Please train the model by 'python train.py'.")
    exit(1)

model.eval()

# 预处理变换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>MNIST数字识别API测试</title>
    </head>
    <body>
        <h1>MNIST数字识别API</h1>
        <p>POST请求到 /predict 可以进行数字识别</p>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 从前端接收图像数据
        data = request.json
        image_data = data['image']
        
        # 解码base64图像数据
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(BytesIO(image_bytes)).convert('L')
        
        # 调整大小为28x28
        image = image.resize((28, 28))
        
        # 应用预处理
        tensor = transform(image).unsqueeze(0).to(device)
        
        # 进行预测
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        
        # 找到最高概率的数字
        predicted_class = int(np.argmax(probabilities))
        
        # 返回结果
        result = {
            'prediction': predicted_class,
            'probabilities': [float(p) for p in probabilities]
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)