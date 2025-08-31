# hog_utils.py
import numpy as np
from skimage.feature import hog
from sklearn.base import BaseEstimator, TransformerMixin

class HOGTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.array([
            hog(img.reshape(28, 28),
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block)
            for img in X
        ])
