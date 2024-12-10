import os
import random
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image
from datetime import datetime

class DataPreparation:
    def __init__(self, original_data_dir, output_dir, sample_class_number):
        self.original_data_dir = original_data_dir  # Path to the directory with 1000 personalities
        self.output_dir = output_dir  # Path to store the new dataset
        self.sample_class_number = sample_class_number  # Number of folders to randomly select
        self.selected_classes = []

    def preprocess_images(self, image_path):
        """
        Preprocesses the image for InceptionV3. 
        Resizes and normalizes the image.
        """
        img = load_img(image_path, target_size=(299, 299))  # InceptionV3 accepts 299x299
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0  # Normalize the image
        return img

    def prepare_data(self):
        """
        Selects `sample_class_number` random folders from the original data
        and creates a new folder with preprocessed images for training.
        """
        # Create output directory if not exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        all_folders = os.listdir(self.original_data_dir)
        selected_folders = random.sample(all_folders, self.sample_class_number)
        self.selected_classes = selected_folders

        for folder in selected_folders:
            folder_path = os.path.join(self.original_data_dir, folder)
            output_folder = os.path.join(self.output_dir, folder)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                processed_image = self.preprocess_images(image_path)
                processed_image_path = os.path.join(output_folder, image_name)
                # Save processed images
                processed_image = processed_image[0]  # Remove batch dimension for saving
                processed_image = (processed_image * 255).astype(np.uint8)  # Denormalize
                img = Image.fromarray(processed_image)
                img.save(processed_image_path)

        return self.selected_classes
