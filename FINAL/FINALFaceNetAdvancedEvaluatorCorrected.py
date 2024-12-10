import numpy as np
import os
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import datetime


class FaceNetAdvancedEvaluatorCorrected:
    def __init__(self, dataset_dir, model):
        self.model = model
        self.embeddings_dir = os.path.join(dataset_dir, "embeddings")
        self.dataset_dir = dataset_dir
        self.test_dir = os.path.join(dataset_dir, "test")
        self.datagen = ImageDataGenerator(
            rotation_range=15,          # Random rotation between -30 and 30 degrees
            width_shift_range=0.2,      # Horizontal shift
            height_shift_range=0.2,     # Vertical shift
            shear_range=0.2,            # Shear transformation
            zoom_range=0.2,             # Random zoom
            horizontal_flip=True,       # Flip images randomly horizontally
            fill_mode='nearest'         # How to fill missing pixels after transformations
        )

    def calculate_similarity(self, test_embedding, candidate_embeddings):
        """
        Calculate cosine similarity between a test embedding and candidate embeddings.
        """
        return [1 - cosine(test_embedding.flatten(), candidate.flatten()) for candidate in candidate_embeddings]

    def predict(self, test_image_path, topN=10):
        """
        Predict the most similar person for a given test image using both average score and frequency score.
        """
        # Load and preprocess test image
        img = load_img(test_image_path, target_size=(299, 299))
        img = img_to_array(img) / 255.0

        # Load embeddings
        all_embeddings = {}
        for file_name in os.listdir(self.embeddings_dir):
            if file_name.endswith("_embedding.npy"):
                person = file_name.split("_embedding")[0]
                all_embeddings[person] = np.load(os.path.join(self.embeddings_dir, file_name))

        

        """
        Compares the test image to the saved embeddings and returns the similarity score.
        """
        print(f"{test_image_path}")
        img = load_img(test_image_path, target_size=(299, 299))
        img = img_to_array(img)  # Convert image to array
        img = img / 255.0 
        augmented_test_images = self.datagen.flow(np.expand_dims(img, axis=0), batch_size=1)

        # Progressive narrowing using binary search style
        remaining_persons = list(all_embeddings.keys())
        print(f'Total Remaining : {len(remaining_persons)}')
        num_iterations = 10
        cumulative_scores = {person: 0 for person in remaining_persons}
        frequency_scores = {person: 0 for person in remaining_persons}

        #For small datasets we need to keep the while loop running for atleast 2 iterations and 10 wont be the correct number
        result = min(int(len(remaining_persons) / 4), topN)
        print(f'Binary Search till {result}')
        currentIter = 0
        while currentIter < num_iterations:

            #For each iteration we augment the test image a little and recalculate the score but this time not with
            #entire dataset but half of it
            # Get embeddings from the test image
            augmented_test_image = next(augmented_test_images)[0]
            test_embedding = self.model.get_embeddings(augmented_test_image).numpy()     
            round_scores = []
            for person in remaining_persons:
                similarities = self.calculate_cosine_similarity(test_embedding, all_embeddings[person])
                round_scores.append((person, max(similarities)))  # Choose the max similarity in this round
            currentIter+=1
            # Sort and keep top N/2
            round_scores.sort(key=lambda x: x[1], reverse=True)
            # Add round scores to cumulative scores
            for index, (person, similarity) in enumerate(round_scores):
                cumulative_scores[person] += similarity
                frequency_scores[person] += (len(round_scores) - index)

        average_scores = {person: cumulative_scores[person] / num_iterations for person in remaining_persons}
        prediction_by_average = sorted(average_scores.items(), key=lambda x: x[1], reverse=True)
        prediction_by_total = sorted(frequency_scores.items(), key=lambda x: x[1], reverse=True)
        return prediction_by_average, prediction_by_total

    def evaluate(self):
        """
        Evaluate the model using the test set and embeddings.
        """
        correct_predictions_avg = 0
        correct_predictions_freq = 0
        total_tests = 0
        distance_frq_based = 0
        distance_avg_based = 0
        max_distance_frq_based = 0
        max_distance_avg_based = 0

        for folder_name in os.listdir(self.test_dir):
            test_folder = os.path.join(self.test_dir, folder_name)
            if os.path.isdir(test_folder):
                for image_name in os.listdir(test_folder):
                    test_image_path = os.path.join(test_folder, image_name)
                    prediction_avg, prediction_freq = self.predict(test_image_path)

                    total_tests += 1
                    if prediction_freq[0][0] == folder_name:
                        correct_predictions_freq += 1
                    if prediction_avg[0][0] == folder_name:
                        correct_predictions_avg += 1
                    for index, (name, score) in enumerate(prediction_avg):
                        if name == folder_name:
                            max_distance_avg_based = max(distance_avg_based, max_distance_avg_based)
                            distance_avg_based += index
                    for index, (name, score) in enumerate(prediction_freq):
                        if name == folder_name:
                            max_distance_frq_based = max(distance_frq_based, max_distance_frq_based)
                            distance_frq_based += index

        accuracy_avg = correct_predictions_avg / total_tests * 100
        accuracy_freq = correct_predictions_freq / total_tests * 100
        avg_distance_avg_based = distance_avg_based / total_tests
        avg_distance_frq_based = distance_frq_based / total_tests

        print(f"Accuracy using average score: {accuracy_avg:.2f}%")
        print(f"Accuracy using frequency score: {accuracy_freq:.2f}%")
        print(f"Average Distance using frequency score: {avg_distance_frq_based:.2f}%")
        print(f"Average Distance using average score: {avg_distance_avg_based:.2f}%")
        metrics = {
            "Accuracy (Average)": accuracy_avg,
            "Accuracy (Frequency)": accuracy_freq,
            "Distance Avg Based": avg_distance_avg_based,
            "Distance Frq Based": avg_distance_frq_based,
            "Max Distance Avg Based" : max_distance_avg_based,
            "Max Distance Frq Based": max_distance_frq_based
        }
        model_name = os.path.basename(self.dataset_dir)
        timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
        # Save results to CSV
        df_results = pd.DataFrame([metrics])
        output_dir = f'../savedmodels/{model_name}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        results_file = f'{output_dir}/evaluation_results_{timestamp}.csv'
        # Create output directory if not exists
        df_results.to_csv(results_file, index=False)

    def calculate_cosine_similarity(self, test_embedding, candidate_embeddings):
            """
            Calculate cosine similarity between a test embedding and candidate embeddings.
            """
            return [1 - cosine(test_embedding.flatten(), candidate.flatten()) for candidate in candidate_embeddings]