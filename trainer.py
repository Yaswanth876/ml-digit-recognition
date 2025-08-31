"""
Train a scikit-learn MNIST classifier using HOG features + LinearSVC (calibrated).
Yields ~97–98% on MNIST and is robust to hand-drawn digits with proper preprocessing.
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from skimage.feature import hog
import joblib
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from hog_utils import HOGTransformer


RANDOM_STATE = 42
MODEL_PATH = "mnist_sklearn_hog_pipeline.pkl"

class HOGTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, orientations=9, pixels_per_cell=(4,4), cells_per_block=(2,2), block_norm="L2-Hys"):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X: (n_samples, 784) in [0,1]
        imgs = X.reshape(-1, 28, 28)
        feats = [
            hog(img,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                block_norm=self.block_norm)
            for img in imgs
        ]
        return np.asarray(feats, dtype=np.float32)

def load_mnist():
    # Prefer pandas parser if available; fall back to liac-arff if not.
    try:
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    except ImportError:
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="liac-arff")
    X = (X / 255.0).astype(np.float32)
    y = y.astype(np.int32)
    return X, y

def main():
    print("🔬 Loading MNIST…")
    X, y = load_mnist()

    # Standard MNIST split: first 60k train, last 10k test
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    print("🚀 Building pipeline: HOG → LinearSVC (calibrated)")
    base = LinearSVC(random_state=RANDOM_STATE)
    clf = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)


    pipe = Pipeline([
        ("hog", HOGTransformer(orientations=9, pixels_per_cell=(4,4), cells_per_block=(2,2), block_norm="L2-Hys")),
        ("clf", clf)
    ])

    print("🏋️ Training… (this may take a few minutes)")
    pipe.fit(X_train, y_train)

    print("📊 Evaluating…")
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"🎯 Test Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, digits=4))

    joblib.dump(pipe, MODEL_PATH)
    print(f"💾 Saved model pipeline → {MODEL_PATH}")

if __name__ == "__main__":
    main()
