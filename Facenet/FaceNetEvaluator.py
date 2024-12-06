import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

class FaceNetEvaluator:
    def __init__(self, dataset_dir, model):
        """
        Initialize the evaluator with the dataset directory.
        The dataset directory contains the 'test' and 'embeddings' subfolders.
        """
        self.model = model
        self.dataset_dir = dataset_dir
        self.embeddings_dir = os.path.join(self.dataset_dir, 'embeddings')
        self.test_dir = os.path.join(self.dataset_dir, 'test')

    def load_embeddings(self):
        """
        Load the embeddings for each class from the embeddings directory.
        """
        embeddings = {}
        for file_name in os.listdir(self.embeddings_dir):
            if file_name.endswith("_embedding.npy"):
                class_name = file_name.split('_')[0]  # Assuming the file format is class_name_embedding.npy
                embedding = np.load(os.path.join(self.embeddings_dir, file_name))
                embeddings[class_name] = embedding
        return embeddings

    def evaluate(self):
        """
        Evaluate the model by comparing test images with stored embeddings.
        """
        embeddings = self.load_embeddings()
        correct_predictions = 0
        total_images = 0

        for folder_name in os.listdir(self.test_dir):
            folder_path = os.path.join(self.test_dir, folder_name)
            if os.path.isdir(folder_path):
                for image_name in os.listdir(folder_path):
                    image_path = os.path.join(folder_path, image_name)
                    # Get embedding for the test image
                    test_embedding = self.preprocess_and_embed(image_path, self.model)
                    # Find the most similar class by cosine similarity
                    max_similarity = -1
                    predicted_class = None
                    for class_name, class_embedding in embeddings.items():
                        similarity = cosine_similarity(test_embedding, class_embedding)
                        if similarity > max_similarity:
                            max_similarity = similarity
                            predicted_class = class_name
                    
                    # Check if the predicted class matches the actual class
                    if predicted_class == folder_name:
                        correct_predictions += 1
                    total_images += 1

        # Calculate accuracy
        accuracy = correct_predictions / total_images * 100
        print(f"Accuracy: {accuracy:.2f}%")

    def preprocess_and_embed(self, image_path, model):
        """
        Preprocesses the image, generates an embedding for the image.
        """
        img = load_img(image_path, target_size=(299, 299))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        img = img / 255.0  # Normalize
        embedding = model.get_embeddings(img)
        return embedding.numpy()