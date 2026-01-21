import sys
import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as transforms
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame, QDialog
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# Configure matplotlib to use a Chinese-capable font when available (Windows)
_available_fonts = set(f.name for f in fm.fontManager.ttflist)
for _name in ("Microsoft YaHei", "SimHei", "Arial Unicode MS", "SimSun"):
    if _name in _available_fonts:
        plt.rcParams['font.sans-serif'] = [_name]
        break
plt.rcParams['axes.unicode_minus'] = False
from PyQt5.QtGui import QPainter, QPen, QColor, QPixmap
from PyQt5.QtCore import Qt, QPoint

from model import Minst_cnn
from utils import get_device

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

device = get_device()
model = Minst_cnn(num_classes=10).to(device)
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

class PaintBoard(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(280, 280)
        self.setFrameShape(QFrame.Box)
        self.setStyleSheet("background-color: black;border: 2px solid gray;")
        self.setAttribute(Qt.WA_StaticContents)
        # persistent pixmap to retain strokes between paint events
        self.pixmap = QPixmap(self.size())
        self.pixmap.fill(QColor(0, 0, 0))
        self.is_drawing = False
        self.lastPoint = QPoint()
        self.pil_img = Image.new("L", (28, 28), 0)
        self.pil_draw = ImageDraw.Draw(self.pil_img)
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_drawing = True
            self.lastPoint = event.pos()

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) & self.is_drawing:
            currentPoint = event.pos()
            # draw onto persistent pixmap so strokes remain visible
            painter = QPainter(self.pixmap)
            pen = QPen(QColor(255, 255, 255), 20, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.lastPoint, currentPoint)
            painter.end()

            # map widget coordinates (280x280) to PIL 28x28
            w, h = self.width(), self.height()
            def to_pil(pt):
                x = int(pt.x() * 28 / w)
                y = int(pt.y() * 28 / h)
                x = max(0, min(27, x))
                y = max(0, min(27, y))
                return x, y

            x1, y1 = to_pil(self.lastPoint)
            x2, y2 = to_pil(currentPoint)
            self.pil_draw.line((x1, y1, x2, y2), fill=255, width=2)
            self.lastPoint = QPoint(currentPoint)
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_drawing = False

    def clear(self):
        # clear pixmap and PIL image
        self.pixmap.fill(QColor(0, 0, 0))
        self.update()
        self.pil_img = Image.new("L", (28, 28), 0)
        self.pil_draw = ImageDraw.Draw(self.pil_img)
    
    def get_img(self):
        return self.pil_img

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.pixmap)
        painter.end()
    
class AppDemo(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("数字识别(PyTorch+PyQt5)")
        self.setFixedSize(400, 450)
        self.layout = QVBoxLayout()
        self.layout.setSpacing(20)
        self.layout.setAlignment(Qt.AlignCenter)

        self.title_label = QLabel("MNIST数字识别")
        self.title_label.setStyleSheet("font-size: 18px;font-weight: bold;")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.title_label)

        self.drawboard = PaintBoard()
        self.layout.addWidget(self.drawboard, alignment=Qt.AlignCenter)

        self.btn_layout = QHBoxLayout()
        self.btn_layout.setSpacing(30)
        self.clear_btn = QPushButton("清屏")
        self.clear_btn.setStyleSheet("font-size: 16px;padding: 8px 20px;")
        self.clear_btn.clicked.connect(self.clear_board)

        self.predict_btn = QPushButton("识别")
        self.predict_btn.setStyleSheet("font-size: 16px;padding: 8px 20px;background-color: lightblue;color: white;border:none;border-radius: 4px;")
        self.predict_btn.clicked.connect(self.predict_digit)

        self.btn_layout.addWidget(self.clear_btn)
        self.btn_layout.addWidget(self.predict_btn)
        self.layout.addLayout(self.btn_layout)

        self.result_label = QLabel("识别结果: ")
        self.result_label.setStyleSheet("font-size: 24px;font-weight: bold;color: blue;")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.result_label)

        self.setLayout(self.layout)
    def clear_board(self):
        self.drawboard.clear()
        self.result_label.setText("识别结果: ")

    def predict_digit(self):
        try:
            img = self.drawboard.get_img()

            img_tensor = transform(img).unsqueeze(0)
            img_tensor = img_tensor.to(device)

            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                pred_idx = int(probs.argmax())
                pred_digit = str(pred_idx)

            self.result_label.setText("识别结果: " + pred_digit)

            # 弹出概率窗口（matplotlib 绘图）
            dlg = ProbDialog(probs, pred_idx, parent=self)
            dlg.exec_()
            # 关闭弹窗后清屏
            self.clear_board()
        except Exception as e:
            self.result_label.setText("识别出错")
            print(e)


class ProbDialog(QDialog):
    def __init__(self, probs, pred_idx, parent=None):
        super().__init__(parent)
        self.setWindowTitle("预测概率")
        self.setFixedSize(480, 360)

        layout = QVBoxLayout()

        title = QLabel(f"识别结果: {pred_idx}")
        title.setStyleSheet("font-size:20px;font-weight:bold;color:green;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # matplotlib Figure
        fig = Figure(figsize=(4.5, 2.5))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        x = list(range(10))
        y = [float(p) * 100.0 for p in probs]
        colors = ["orange" if i == pred_idx else "skyblue" for i in x]
        bars = ax.bar(x, y, color=colors)
        ax.set_xticks(x)
        ax.set_xlabel("Digit")
        ax.set_ylabel("Probability (%)")
        ax.set_ylim(0, max(100.0, max(y) * 1.1))
        ax.set_title("预测概率分布")

        # annotate percentages above bars
        for bar, val in zip(bars, y):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, h + 1, f"{val:.1f}%", ha='center', va='bottom', fontsize=9)

        fig.tight_layout()
        layout.addWidget(canvas)

        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignCenter)

        self.setLayout(layout)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    demo = AppDemo()
    demo.show()
    sys.exit(app.exec_())