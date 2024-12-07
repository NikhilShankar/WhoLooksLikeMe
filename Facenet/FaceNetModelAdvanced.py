import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#This model saves embeddings for all images and does a binary search to find similarities 
#Other models rely on saving average embedding
class FaceNetModelAdvanced:
    def __init__(self, model_dir="../inception-v3"):
        """
        Load the InceptionResNetV1 model saved in TensorFlow's SavedModel format.
        """
        self.model = tf.saved_model.load(model_dir)
        self.model.summary()
        # Define the ImageDataGenerator for augmentations
        self.datagen = ImageDataGenerator(
            rotation_range=30,          # Random rotation between -30 and 30 degrees
            width_shift_range=0.2,      # Horizontal shift
            height_shift_range=0.2,     # Vertical shift
            shear_range=0.2,            # Shear transformation
            zoom_range=0.2,             # Random zoom
            horizontal_flip=True,       # Flip images randomly horizontally
            fill_mode='nearest'         # How to fill missing pixels after transformations
        )
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

    def create_embeddings_for_personality(self, data_dir, should_augment: True, augmentation_count = 5):
        """
        For each folder (personality), create embeddings and save them to the embeddings folder.
        """
        print("Augmentation : {should_augment} with augmentated image count : {augmentation_count}")
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
                    if should_augment:
                        # Apply augmentation
                        augmented_images = self.datagen.flow(np.expand_dims(img, axis=0), batch_size=1)

                        # Process each augmented image and get embeddings
                        augmented_embeddings = []
                        for _ in range(augmentation_count):  # You can control the number of augmented images to generate
                            augmented_img = next(augmented_images)[0]
                            embedding = self.get_embeddings(augmented_img)
                            augmented_embeddings.append(embedding)
                        # Average embeddings for this person (you can also use other methods like max)
                        embedding = np.mean(np.array(augmented_embeddings), axis=0)
                    else:
                        embedding = self.get_embeddings(img)
                    folder_embeddings.append(embedding)

                ##IMPORTANT We are removing averaging of embeddings. Instead we will be saving each embeddings separately.
                # Average embeddings for each person (you can also try other methods like max)
                #avg_embedding = np.mean(np.array(folder_embeddings), axis=0)
                #embeddings[folder_name] = avg_embedding
                embeddings[folder_name] = np.array(folder_embeddings)

                # Save the embeddings to a file inside the embeddings folder
                embedding_file = os.path.join(embeddings_dir, f"{folder_name}_embedding.npy")
                np.save(embedding_file, np.array(folder_embeddings))

        return embeddings
