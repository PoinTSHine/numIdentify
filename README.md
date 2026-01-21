# num_identify

一个用 PyTorch + PyQt5 实现的 MNIST 手写数字识别示例 GUI 程序。

主要文件
- `gui.py`：主 GUI 程序，包含绘制面板和预测/概率可视化。
- `model.py`：模型定义（`Minst_cnn`）。
- `train.py`：训练脚本。
- `utils.py`：辅助函数（设备选择、数据加载等）。

运行说明
1. 创建并激活 Python 环境（例如 conda/venv）。
2. 安装依赖（见 `requirements.txt`）：

```bash
pip install -r requirements.txt
```

3. 把训练好的模型文件 `mnist_cnn.pth` 放在项目根目录（该文件不包含在仓库内）。
4. 运行 GUI：

```bash
python gui.py
```
