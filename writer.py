"""
Interactive MNIST Digit Recognition (scikit-learn pipeline)
- Proper MNIST-style preprocessing: crop → resize(20) → pad(28) → center → normalize
- Uses the calibrated LinearSVC+HOG pipeline saved by trainer.py
"""

import tkinter as tk
import numpy as np
import cv2
from PIL import Image, ImageDraw
import joblib
from hog_utils import HOGTransformer

MODEL_PATH = "mnist_sklearn_hog_pipeline.pkl"

class MNISTInteractiveDemo:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🎯 scikit-learn Digit Recognition")
        self.root.geometry("820x620")
        self.canvas_width, self.canvas_height = 280, 280

        # Load trained pipeline
        try:
            self.model = joblib.load(MODEL_PATH)
            print("✅ Model pipeline loaded!")
            self.model_loaded = True
        except Exception as e:
            print(f"❌ Model not found! Run trainer.py first. Details: {e}")
            self.model_loaded = False

        # Drawing buffer
        self.drawing = False
        self.last_x, self.last_y = None, None
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image)

        self.setup_ui()

    def setup_ui(self):
        left = tk.Frame(self.root)
        left.pack(side="left", padx=20, pady=20)

        self.canvas = tk.Canvas(left, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_on_canvas)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

        btns = tk.Frame(left)
        btns.pack(pady=10)
        tk.Button(btns, text="🗑 Clear", command=self.clear_canvas, width=10).pack(side="left", padx=5)
        tk.Button(btns, text="🔮 Predict", command=self.predict_digit, width=10).pack(side="left", padx=5)

        right = tk.Frame(self.root)
        right.pack(side="right", padx=20, pady=20)
        self.result_label = tk.Label(right, text="Draw a digit (0–9)", font=("Arial", 20))
        self.result_label.pack()

    def start_draw(self, event):
        self.drawing = True
        self.last_x, self.last_y = event.x, event.y

    def draw_on_canvas(self, event):
        if self.drawing:
            # Slightly thinner stroke helps resemble MNIST
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, width=12, fill="black", capstyle=tk.ROUND)
            self.draw.line([self.last_x, self.last_y, event.x, event.y], fill="black", width=12)
            self.last_x, self.last_y = event.x, event.y

    def stop_draw(self, event):
        self.drawing = False

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="Draw a digit (0–9)", fg="black")

    # --- MNIST-style preprocessing ---
    @staticmethod
    def preprocess_to_28x28(img_280x280: np.ndarray) -> np.ndarray:
        """
        Convert canvas image (white bg, black ink) to MNIST-like 28x28 float [0,1]
        Steps: invert -> binarize -> crop bbox -> resize longest side to 20 -> pad to 28 -> center by moments
        """
        # 1) Invert: black ink -> white foreground (like MNIST)
        img = 255 - img_280x280.astype(np.uint8)

        # 2) Binarize (Otsu)
        _, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # 3) Crop to bounding box of ink
        coords = cv2.findNonZero(bin_img)
        if coords is None:
            return np.zeros((28, 28), dtype=np.float32)
        x, y, w, h = cv2.boundingRect(coords)
        digit = bin_img[y:y+h, x:x+w]

        # 4) Resize preserving aspect ratio so longer side = 20
        if w > h:
            new_w, new_h = 20, max(1, int(round(h * 20.0 / w)))
        else:
            new_h, new_w = 20, max(1, int(round(w * 20.0 / h)))
        digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 5) Pad to 28x28
        pad_left  = (28 - new_w) // 2
        pad_right = 28 - new_w - pad_left
        pad_top   = (28 - new_h) // 2
        pad_bot   = 28 - new_h - pad_top
        digit = cv2.copyMakeBorder(digit, pad_top, pad_bot, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)

        # 6) Center via image moments
        M = cv2.moments(digit)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            shift_x = int(round(14 - cx))
            shift_y = int(round(14 - cy))
            M_shift = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            digit = cv2.warpAffine(digit, M_shift, (28, 28), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # 7) Normalize to [0,1] floats
        return (digit.astype(np.float32) / 255.0)

    def predict_digit(self):
        if not self.model_loaded:
            self.result_label.config(text="❌ Model not loaded!", fg="red")
            return

        canvas_img = np.array(self.image)  # 280x280, white bg
        img28 = self.preprocess_to_28x28(canvas_img)
        x = img28.reshape(1, -1)  # (1,784)

        # Pipeline handles HOG + classifier internally
        probs = None
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(x)[0]
            pred = int(np.argmax(probs))
            conf = float(np.max(probs)) * 100.0
        else:
            pred = int(self.model.predict(x)[0])
            conf = 0.0

        if probs is not None:
            self.result_label.config(
                text=f"🎯 Prediction: {pred}\nConfidence: {conf:.2f}%",
                fg="green" if conf >= 85 else "orange"
            )
        else:
            self.result_label.config(text=f"🎯 Prediction: {pred}\n(no calibrated confidence)", fg="black")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = MNISTInteractiveDemo()
    app.run()
