from scipy.spatial.distance import cosine
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array


class SimilarityCalculator:
    def __init__(self, embeddings_dir):
        self.embeddings_dir = embeddings_dir
        self.embeddings = self.load_embeddings()

    def load_embeddings(self):
        """
        Loads the saved embeddings from files.
        """
        embeddings = {}
        for embedding_file in os.listdir(self.embeddings_dir):
            if embedding_file.endswith("_embedding.npy"):
                folder_name = embedding_file.replace("_embedding.npy", "")
                embedding_path = os.path.join(self.embeddings_dir, embedding_file)
                embedding = np.load(embedding_path)
                embeddings[folder_name] = embedding
        return embeddings

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

    def calculate_similarity(self, test_image_path, model):
        """
        Compares the test image to the saved embeddings and returns the similarity score.
        """
        test_embedding = self.preprocess_and_embed(test_image_path, model)
        
        similarities = {}
        for folder_name, embedding in self.embeddings.items():
            flattened_test_embedding = test_embedding.flatten()
            flattened_embedding = embedding.flatten()
            similarity_score = 1 - cosine(flattened_test_embedding, flattened_embedding)  # Cosine similarity
            similarities[folder_name] = similarity_score
        
        # Sort by highest similarity
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return sorted_similarities
