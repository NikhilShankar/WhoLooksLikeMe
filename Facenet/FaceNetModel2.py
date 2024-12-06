import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class FaceNetModel2:
    def __init__(self, model_dir="../inception-v3"):
        """
        Load the InceptionResNetV1 model saved in TensorFlow's SavedModel format.
        """
        self.model = tf.saved_model.load(model_dir)
        print("InceptionResNetV1 model loaded successfully!")

    def get_embeddings(self, image):
        # Ensure the image has the correct shape (1, 299, 299, 3)
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)  # Add batch dimension if missing
        """
        Accepts a preprocessed image and returns the embedding.
        """
        embedding = self.model(image)  # Get embeddings from the model
        return embedding

    def create_embeddings_for_personality(self, data_dir):
        """
        For each folder (personality), create embeddings and save them to the embeddings folder.
        """
        input_dir = f'{data_dir}/train'
        embeddings_dir = f'{data_dir}/embeddings'
        embeddings = {}
        if not os.path.exists(embeddings_dir):
            os.makedirs(embeddings_dir)

        for folder_name in os.listdir(input_dir):
            folder_path = os.path.join(input_dir, folder_name)
            if os.path.isdir(folder_path):
                folder_embeddings = []
                for image_name in os.listdir(folder_path):
                    image_path = os.path.join(folder_path, image_name)
                    img = load_img(image_path, target_size=(299, 299))  # InceptionV3 input size
                    img = img_to_array(img)  # Convert image to array
                    img = img / 255.0  # Normalize image to [0, 1]
                    
                    # Get embedding for each image
                    embedding = self.get_embeddings(img)
                    folder_embeddings.append(embedding)

                # Average embeddings for each person (you can also try other methods like max)
                avg_embedding = np.mean(np.array(folder_embeddings), axis=0)
                embeddings[folder_name] = avg_embedding

                # Save the embeddings to a file inside the embeddings folder
                embedding_file = os.path.join(embeddings_dir, f"{folder_name}_embedding.npy")
                np.save(embedding_file, avg_embedding)

        return embeddings
