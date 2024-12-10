import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
import datetime
import pandas as pd


#This model saves embeddings for all images and does a binary search to find similarities 
#Other models rely on saving average embedding
class FaceNetModelAdvanced:
    def __init__(self, model_dir="../inception-v3", dataset_dir):
        """
        Initializes the model with a dataset directory and sets the number of classes.
        """
        self.dataset_dir = dataset_dir
        self.train_dir = os.path.join(dataset_dir, 'train')
        self.test_dir = os.path.join(dataset_dir, 'test')

        # Automatically determine number of classes from the dataset folder structure
        self.class_names = sorted(os.listdir(self.train_dir))
        self.num_classes = len(self.class_names)

        # Base model (InceptionResNetV2 or any other pre-trained model)
        #self.base_model = load_model(model_dir)
        #model = load_model('c:\Users\DELL\.cache\kagglehub\models\gannayasser\facenet\keras\default\1\facenet_keras.h5')
        #model = hub.load('https://tfhub.dev/google/facenet/1')
        base_model = InceptionV3(weights="imagenet", include_top=False, pooling="avg")


        # Add a pooling layer to reduce the output size
        x = base_model.output

        # Add a dense layer for classification (softmax output for n classes)
        class_output = Dense(self.num_classes, activation='softmax', name='classification')(x)

        # Add a dense layer for generating embeddings (linear activation)
        #embedding_output = Dense(embedding_dimension, activation='linear', name='embedding_layer')(x)

        # Create the model
        print(base_model.input)
        self.model = Model(inputs=base_model.input, outputs=class_output)

        # Freeze the base model (optional, for fine-tuning later)
        for layer in base_model.layers:
            layer.trainable = False

        # Compile the model
        #self.model.compile(optimizer=Adam(), loss={'classification': 'categorical_crossentropy', 'embedding_layer': 'mean_squared_error'}, metrics=['accuracy'])
        self.model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

        print(self.model.summary())
        """
        Load the InceptionResNetV1 model saved in TensorFlow's SavedModel format.
        """
        self.model = tf.saved_model.load(model_dir)
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
        start_train_time = time.time()
        self.dataset_dir = data_dir
        input_dir = f'{data_dir}/train'
        embeddings_dir = f'{data_dir}/embeddings'
        embeddings = {}
        if not os.path.exists(embeddings_dir):
            os.makedirs(embeddings_dir)

        for folder_name in os.listdir(input_dir):
            folder_path = os.path.join(input_dir, folder_name)
            if os.path.isdir(folder_path):
                folder_embeddings = []
                maxCount = 0
                for image_name in os.listdir(folder_path):
                    if(maxCount == 3):
                        break
                    maxCount+=1
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
                    print(f"Details of embedding : Shape: {embedding.shape}")
                    folder_embeddings.append(embedding)

                ##IMPORTANT We are removing averaging of embeddings. Instead we will be saving each embeddings separately.
                # Average embeddings for each person (you can also try other methods like max)
                #avg_embedding = np.mean(np.array(folder_embeddings), axis=0)
                #embeddings[folder_name] = avg_embedding
                embeddings[folder_name] = np.array(folder_embeddings)

                # Save the embeddings to a file inside the embeddings folder
                embedding_file = os.path.join(embeddings_dir, f"{folder_name}_embedding.npy")
                np.save(embedding_file, np.array(folder_embeddings))
        train_time = time.time() - start_train_time
        metrics = {
            "Train Time": train_time
        }
        model_name = os.path.basename(self.dataset_dir)
        timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
        # Save results to CSV
        df_results = pd.DataFrame([metrics])
        output_dir = f'../savedmodels/{model_name}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        results_file = f'{output_dir}/training_dir_{timestamp}.csv'
        df_results.to_csv(results_file, index=False)
        return embeddings
