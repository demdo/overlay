import sys
import numpy as np
import cv2
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QScrollArea, QSizePolicy
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt


def load_uv(uv_path):
    ext = uv_path.split(".")[-1].lower()
    if ext == "npy":
        uv = np.load(uv_path)
    elif ext == "npz":
        data = np.load(uv_path)
        for key in ["uv", "pts", "points", "keypoints", "UV"]:
            if key in data:
                uv = data[key]
                break
        else:
            key = list(data.keys())[0]
            print(f"[INFO] NPZ: using first key '{key}'")
            uv = data[key]
    elif ext in ["txt", "csv"]:
        delim = "," if ext == "csv" else None
        uv = np.loadtxt(uv_path, delimiter=delim)
    else:
        raise ValueError(f"Unsupported UV format: .{ext}")

    uv = np.array(uv, dtype=np.float32)
    if uv.ndim == 1:
        uv = uv.reshape(-1, 2)
    return uv[:, :2]


def draw_crosses(img, uv, color=(0, 255, 0), size=14, thickness=2):
    out = img.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    elif out.shape[2] == 4:
        out = cv2.cvtColor(out, cv2.COLOR_BGRA2BGR)

    h, w = out.shape[:2]

    for i, (u, v) in enumerate(uv):
        x, y = int(round(u)), int(round(v))

        if not (0 <= x < w and 0 <= y < h):
            print(f"[WARN] Point {i} ({x},{y}) outside image ({w}x{h})")
            continue

        # cross
        cv2.line(out, (x - size, y), (x + size, y), color, thickness)
        cv2.line(out, (x, y - size), (x, y + size), color, thickness)

        # number label
        label = str(i)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.55
        ft = 1
        (tw, th), _ = cv2.getTextSize(label, font, fs, ft)
        tx, ty = x + size + 4, y + th // 2
        cv2.rectangle(out, (tx - 1, ty - th - 1), (tx + tw + 2, ty + 2), (0, 0, 0), -1)
        cv2.putText(out, label, (tx, ty), font, fs, color, ft, cv2.LINE_AA)

    return out


def cv2_to_qpixmap(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = img_rgb.shape
    qimg = QImage(img_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("X-ray UV Debug Viewer")
        self.resize(1000, 750)

        self.xray_img = None
        self.uv_pts   = None

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # buttons row
        btn_row = QHBoxLayout()

        self.btn_xray = QPushButton("📂  Load X-ray Image")
        self.btn_xray.clicked.connect(self.load_xray)

        self.btn_uv = QPushButton("📂  Load UV File")
        self.btn_uv.clicked.connect(self.load_uv_file)
        self.btn_uv.setEnabled(False)

        self.btn_save = QPushButton("💾  Save Result")
        self.btn_save.clicked.connect(self.save_result)
        self.btn_save.setEnabled(False)

        self.lbl_status = QLabel("Load an X-ray image to start.")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        btn_row.addWidget(self.btn_xray)
        btn_row.addWidget(self.btn_uv)
        btn_row.addWidget(self.btn_save)
        btn_row.addStretch()
        btn_row.addWidget(self.lbl_status)
        main_layout.addLayout(btn_row)

        # image display
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.img_label = QLabel("No image loaded.")
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.scroll.setWidget(self.img_label)
        main_layout.addWidget(self.scroll, stretch=1)

    def load_xray(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open X-ray Image", "",
            "Images (*.png *.jpg *.jpeg *.tiff *.tif *.bmp);;All Files (*)"
        )
        if not path:
            return

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            self.lbl_status.setText(f"❌ Could not load: {path}")
            return

        if img.dtype == np.uint16:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        self.xray_img = img
        self.uv_pts   = None
        self.btn_uv.setEnabled(True)
        self.btn_save.setEnabled(False)

        self._show_image(img)
        self.lbl_status.setText(
            f"✅ Image: {img.shape[1]}×{img.shape[0]}  |  {path.split('/')[-1]}"
        )

    def load_uv_file(self):
        if self.xray_img is None:
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "Open UV File", "",
            "UV Files (*.npy *.npz *.txt *.csv);;All Files (*)"
        )
        if not path:
            return

        try:
            uv = load_uv(path)
        except Exception as e:
            self.lbl_status.setText(f"❌ UV load error: {e}")
            return

        self.uv_pts = uv
        self.btn_save.setEnabled(True)

        result = draw_crosses(self.xray_img, uv)
        self._show_image(result)
        self.lbl_status.setText(
            f"✅ {uv.shape[0]} points  |  {path.split('/')[-1]}"
        )
        print(f"[INFO] {uv.shape[0]} UV points:")
        for i, (u, v) in enumerate(uv):
            print(f"  [{i:3d}]  u={u:.2f}  v={v:.2f}")

    def save_result(self):
        if self.xray_img is None or self.uv_pts is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Result Image", "result_uv_debug.png",
            "PNG (*.png);;JPEG (*.jpg);;All Files (*)"
        )
        if not path:
            return

        result = draw_crosses(self.xray_img, self.uv_pts)
        cv2.imwrite(path, result)
        self.lbl_status.setText(f"💾 Saved: {path.split('/')[-1]}")

    def _show_image(self, img_bgr):
        pixmap = cv2_to_qpixmap(img_bgr)
        scaled = pixmap.scaled(
            self.scroll.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.img_label.setPixmap(scaled)
        self.img_label.resize(scaled.size())


if __name__ == "__main__":
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())