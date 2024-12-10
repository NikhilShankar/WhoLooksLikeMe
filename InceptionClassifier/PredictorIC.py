import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class PredictorIC:
    def __init__(self, model, image_size=(299, 299)):
        self.model = model
        self.image_size = image_size

    def predict(self, image_path, top_x=5):
        img = load_img(image_path, target_size=self.image_size)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = self.model.predict(img_array)
        top_indices = np.argsort(predictions[0])[::-1][:top_x]
        return top_indices, predictions[0][top_indices]
