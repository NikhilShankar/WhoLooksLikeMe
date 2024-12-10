import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import os

class PredictorVGG:
    def __init__(self, model, train_dir, image_size=(224, 224)):
        self.model = model
        self.image_size = image_size
        
        # Sort class labels from the train directory
        self.class_labels = sorted(os.listdir(train_dir))  # Alphabetically sorted folder names

    def predict(self, image_path, top_x=5):
        # Load and preprocess image
        img = load_img(image_path, target_size=self.image_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Preprocess the input for VGG16
        img_array = preprocess_input(img_array)

        # Get predictions
        predictions = self.model.predict(img_array)

        # Get top X predicted class indices
        top_indices = np.argsort(predictions[0])[::-1][:top_x]
        
        # Map top indices to class labels
        top_labels = [self.class_labels[i] for i in top_indices]
        top_confidences = predictions[0][top_indices]

        return top_labels, top_confidences
