import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input  # Import VGG16 preprocessing
import matplotlib.pyplot as plt

class TrainAndTestVGG:
    def __init__(self, dataset_dir, model, batch_size=32, image_size=(224, 224)):
        self.dataset_dir = dataset_dir
        self.model = model
        self.batch_size = batch_size
        self.image_size = image_size

    def train_model(self, epochs=10):
        train_dir = os.path.join(self.dataset_dir, 'train')
        test_dir = os.path.join(self.dataset_dir, 'test')

        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,  # Normalize the images to [0, 1]
            rotation_range=20,
            zoom_range=0.2,
            horizontal_flip=True,
            preprocessing_function=preprocess_input  # Apply VGG16-specific preprocessing
        )
        
        # Validation / Testing data should only be rescaled, no augmentation
        test_datagen = ImageDataGenerator(
            rescale=1./255,
            preprocessing_function=preprocess_input  # Apply VGG16-specific preprocessing
        )

        # Flow from directory to read images and their labels
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.image_size,  # Resize images to fit VGG16 input size
            batch_size=self.batch_size,
            class_mode='categorical'  # For multi-class classification
        )

        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',  # Multi-class classification
            shuffle=False  # No shuffle for test data to match predictions with true labels
        )

        # Fit the model using the training data and validate on test data
        history = self.model.fit(
            train_generator,
            validation_data=test_generator,
            epochs=epochs,
            verbose=1
        )
        return history

    def evaluate_model(self):
        test_dir = os.path.join(self.dataset_dir, 'test')

        # Testing data preprocessing (only rescaling and BGR conversion for VGG16 input)
        test_datagen = ImageDataGenerator(
            rescale=1./255,
            preprocessing_function=preprocess_input  # Apply VGG16-specific preprocessing
        )
        
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=self.image_size,  # Resize images for VGG16
            batch_size=1,  # For per-image prediction
            class_mode='categorical',  # Multi-class labels
            shuffle=False  # Keep order for evaluating predictions
        )

        # Predictions
        predictions = self.model.predict(test_generator)
        predicted_classes = np.argmax(predictions, axis=1)  # Get the class with max probability
        true_classes = test_generator.classes  # Actual class labels
        class_labels = list(test_generator.class_indices.keys())  # Class names

        # Classification Report
        report = classification_report(true_classes, predicted_classes, target_names=class_labels)
        cm = confusion_matrix(true_classes, predicted_classes)

        print("Classification Report:\n", report)

        # Plot Confusion Matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.show()
