from WLLMModelLoader import WLLMModelLoader
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import seaborn as sns
import os
import math


class ClassificationModel:

    def __init__(self, modelpath, class_names_csv_path):
        #referencing the basic model for classification
        self.modelpath = modelpath
        self.model = WLLMModelLoader(modelpath).model
        df = pd.read_csv(class_names_csv_path)
        self.name_df = df

    def predict_single_image(self, image_path):
        """
        Predict the class of a single image using the classification head.
        """
        img = load_img(image_path, target_size=(299, 299))  # InceptionV3 input size
        img = img_to_array(img)  # Convert image to array
        img = img / 255.0  # Normalize image to [0, 1]
        # Expand dimensions to match the batch shape expected by the model
        img = np.expand_dims(img, axis=0)
        # Get prediction from the classification head
        predictions = self.model.predict(img)  
        # Get the predicted class index
        predicted_class_idx = np.argmax(predictions)
        # Get the class name (folder name) from class_names
        predicted_class_name = self.name_df.iloc[predicted_class_idx]['Class Name']
        result = self.get_sorted_predictions(predictions=predictions[0], class_names_df=self.name_df)
        return predicted_class_name, result
    

    def get_sorted_predictions(self, predictions, class_names_df):
        # Assuming 'Class Name' column exists in class_names_df and predictions is a list/array
        # Create a new DataFrame with 'Class Name' and 'Prediction Score'
        results_df = class_names_df.copy()
        results_df['Prediction Score'] = predictions
        
        # Sort the results by 'Prediction Score' in descending order
        sorted_results_df = results_df.sort_values(by='Prediction Score', ascending=False).reset_index(drop=True)
        
        return sorted_results_df
    

    def evaluate_on_test_folder(self, test_folder_path):
        # Initialize ImageDataGenerator for test data
        test_datagen = ImageDataGenerator(rescale=1.0/255.0)
        
        # Create a generator for the test images
        test_generator = test_datagen.flow_from_directory(
            test_folder_path,
            target_size=(299, 299),  # InceptionV3 input size
            batch_size=4,
            class_mode='categorical',
            shuffle=False  # We don't shuffle for confusion matrix
        )

        # Get the true labels (class indices)
        true_labels = test_generator.classes
        # Map class indices to class names
        class_names = test_generator.class_indices
        class_names = {v: k for k, v in class_names.items()}
        
        # Get the predictions from the model
        predictions = self.model.predict(test_generator, steps=test_generator.samples // test_generator.batch_size, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Calculate confusion matrix
        cm = confusion_matrix(true_labels, predicted_classes)

        # Plot confusion matrix
        plt.figure(figsize=(24, 24))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(class_names.values()), yticklabels=list(class_names.values()))
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=90)
        plt.tight_layout()
        parent_folder = os.path.dirname(self.modelpath)

        plt.savefig(f'{parent_folder}/confusion_matrix.png')  # Save the image
        plt.close()

        # Classification report for accuracy, precision, and recall
        report = classification_report(true_labels, predicted_classes, target_names=list(class_names.values()), output_dict=True)

        # Convert the classification report to a DataFrame
        report_df = pd.DataFrame(report).transpose()

        return report_df
    

    def evaluate_and_print_images(self, test_folder_path):
        # Initialize ImageDataGenerator for test data
        test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

        # Create a generator for the test images
        test_generator = test_datagen.flow_from_directory(
            test_folder_path,
            target_size=(299, 299),  # InceptionV3 input size
            batch_size=4,
            class_mode='categorical',
            shuffle=False  # We don't shuffle for confusion matrix
        )

        # Get the true labels (class indices)
        true_labels = test_generator.classes
        # Map class indices to class names
        class_names = test_generator.class_indices
        class_names = {v: k for k, v in class_names.items()}

        # Get the predictions from the model
        predictions = self.model.predict(
            test_generator, 
            steps=math.ceil(test_generator.samples / test_generator.batch_size), 
            verbose=1
        )
        predicted_classes = np.argmax(predictions, axis=1)

        # Calculate confusion matrix
        cm = confusion_matrix(true_labels, predicted_classes)

        # Plot confusion matrix
        plt.figure(figsize=(24, 24))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(class_names.values()), 
            yticklabels=list(class_names.values())
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=90)
        plt.tight_layout()
        parent_folder = os.path.dirname(self.modelpath)

        plt.savefig(f'{parent_folder}/confusion_matrix.png')  # Save the image
        plt.close()

        # Classification report for accuracy, precision, recall, and F1 score
        report = classification_report(
            true_labels, 
            predicted_classes, 
            target_names=list(class_names.values()), 
            output_dict=True
        )

        # Convert the classification report to a DataFrame
        report_df = pd.DataFrame(report).transpose()

        # Add accuracy per class
        report_df['accuracy'] = 0.0  # Placeholder for accuracy
        for i, class_name in enumerate(class_names.values()):
            true_positive = cm[i, i]
            total_samples = cm[i, :].sum()  # Total samples for this class
            report_df.loc[class_name, 'accuracy'] = true_positive / total_samples if total_samples > 0 else 0.0

        # Save classification report as CSV
        report_df.to_csv(f'{parent_folder}/metrics_test_evaluation.csv', index=True)

        # Collect wrongly predicted image paths
        wrong_predictions = np.where(true_labels != predicted_classes)[0]
        wrong_image_paths = [test_generator.filepaths[i] for i in wrong_predictions]

        # Plot wrongly predicted images
        rows = math.ceil(len(wrong_image_paths) / 10)  # 10 images per row
        fig, axes = plt.subplots(rows, 10, figsize=(20, 2.5 * rows))
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            if i < len(wrong_image_paths):
                img = plt.imread(wrong_image_paths[i])
                true_class = class_names[true_labels[wrong_predictions[i]]]
                pred_class = class_names[predicted_classes[wrong_predictions[i]]]
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(f'True: {true_class}\nPred: {pred_class}', fontsize=8)
            else:
                ax.axis('off')  # Hide extra subplots

        plt.tight_layout()
        plt.savefig(f'{parent_folder}/wrong_predictions.png')
        plt.show()

        # Return metrics
        return report_df[['accuracy', 'precision', 'recall', 'f1-score', 'support']]
