import os
import numpy as np
import time
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime


class TrainAndTestIC:
    def __init__(self, dataset_dir, model, batch_size=32, image_size=(299, 299)):
        self.dataset_dir = dataset_dir
        self.model = model
        self.batch_size = batch_size
        self.image_size = image_size

    def train_model(self, epochs=30):
        train_dir = os.path.join(self.dataset_dir, 'train')
        test_dir = os.path.join(self.dataset_dir, 'test')

        # Data augmentation for training
        train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
        test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            train_dir, target_size=self.image_size, batch_size=self.batch_size, class_mode='categorical'
        )
        test_generator = test_datagen.flow_from_directory(
            test_dir, target_size=self.image_size, batch_size=self.batch_size, class_mode='categorical', shuffle=False
        )

        # Create a ModelCheckpoint callback to save the model during training
        model_name = os.path.basename(self.dataset_dir)
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            f'../savedmodels/{model_name}/best_model.keras',  # File where the model will be saved
            save_best_only=True,  # Save only the best model based on validation accuracy
            monitor='val_loss',  # Monitor validation loss (or accuracy)
            mode='min',  # Save the model with the lowest validation loss
            verbose=1
        )
        start_time = time.time()
        # Fit the model
        history = self.model.fit(
            train_generator,
            validation_data=test_generator,
            epochs=epochs,
            verbose=1,
            callbacks=[checkpoint_callback],
        )
        end_time = time.time()
        training_time = end_time - start_time

        # Evaluate the model on the test set
        start_time_test = time.time()
        test_generator.reset()
        predictions = self.model.predict(test_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_generator.classes
        end_time_test = time.time()
        testing_time = end_time_test - start_time_test

        # Calculate accuracy, precision, recall
        accuracy = accuracy_score(true_classes, predicted_classes)
        precision = precision_score(true_classes, predicted_classes, average='weighted')
        recall = recall_score(true_classes, predicted_classes, average='weighted')
        f1 = f1_score(true_classes, predicted_classes, average='weighted')

        # Create a DataFrame to save the results
        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "training-time": training_time,
            "testing-time": testing_time
        }
        timestamp = datetime.now().strftime("%m-%d-%H-%M")
        # Save results to CSV
        df_results = pd.DataFrame([results])
        results_file = f'../savedmodels/{model_name}/training_results_{timestamp}.csv'
        df_results.to_csv(results_file, index=False)

        print(f"Training complete. Results saved to {results_file}")

        return history

    def evaluate_model(self):
        test_dir = os.path.join(self.dataset_dir, 'test')

        test_datagen = ImageDataGenerator(rescale=1./255)
        test_generator = test_datagen.flow_from_directory(
            test_dir, target_size=self.image_size, batch_size=1, class_mode='categorical', shuffle=False
        )

        # Predictions

        predictions = self.model.predict(test_generator)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_generator.classes
        class_labels = list(test_generator.class_indices.keys())

        # Metrics
        report = classification_report(true_classes, predicted_classes, target_names=class_labels)
        cm = confusion_matrix(true_classes, predicted_classes)

        print("Classification Report:\n", report)

        # Plot confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
        disp.plot(cmap=plt.cm.Blues)
        plt.show()
