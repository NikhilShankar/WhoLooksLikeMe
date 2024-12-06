import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import img_to_array, load_img


class FaceNetModel:
    def __init__(self, model_path="../inception-v3"):
         self.model = tf.saved_model.load(model_path)
         print("InceptionV3 model loaded successfully!")

    def get_embeddings(self, image):
        # Ensure the image has the correct shape (1, 299, 299, 3)
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)  # Add batch dimension if missing
        
        # Check the shape of the image
        print("Image shape after preprocessing:", image.shape)
        """
        Accepts a preprocessed image and returns the embedding.
        """
        embedding = self.model(image)  # Get embeddings from the model
        return embedding

    def create_embeddings_for_personality(self, input_dir, output_dir):
        """
        For each folder (personality), create embeddings and save them to a file.
        """
        embeddings = {}
        for folder_name in os.listdir(input_dir):
            folder_path = os.path.join(input_dir, folder_name)
            if os.path.isdir(folder_path):
                folder_embeddings = []
                for image_name in os.listdir(folder_path):
                    image_path = os.path.join(folder_path, image_name)
                    img = load_img(image_path, target_size=(299, 299))  # InceptionV3 input size
                    img = img_to_array(img)
                    img = np.expand_dims(img, axis=0)  # Add batch dimension
                    img = img / 255.0  # Normalize
                    embedding = self.get_embeddings(img)
                    folder_embeddings.append(embedding)

                # Average embeddings for each person (you can also try other methods like max)
                avg_embedding = np.mean(np.array(folder_embeddings), axis=0)
                embeddings[folder_name] = avg_embedding

                # Save the embeddings to a file
                embedding_file = os.path.join(output_dir, f"{folder_name}_embedding.npy")
                np.save(embedding_file, avg_embedding)
        
        return embeddings
