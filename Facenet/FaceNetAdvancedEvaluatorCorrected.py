import numpy as np
import os
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class FaceNetAdvancedEvaluatorCorrected:
    def __init__(self, dataset_dir, model):
        self.model = model
        self.embeddings_dir = os.path.join(dataset_dir, "embeddings")
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
        from tensorflow.keras.preprocessing.image import load_img, img_to_array
        img = load_img(test_image_path, target_size=(299, 299))
        img = img_to_array(img) / 255.0

        # Load embeddings
        all_embeddings = {}
        for file_name in os.listdir(self.embeddings_dir):
            if file_name.endswith("_embedding.npy"):
                person = file_name.split("_embedding")[0]
                all_embeddings[person] = np.load(os.path.join(self.embeddings_dir, file_name))

        

        # Progressive narrowing using binary search style
        remaining_persons = list(all_embeddings.keys())
        print(f'Total Remaining : {len(remaining_persons)}')
        num_iterations = 0
        scores = {person: 0 for person in remaining_persons}

        #For small datasets we need to keep the while loop running for atleast 2 iterations and 10 wont be the correct number
        result = min(int(len(remaining_persons) / 4), topN)
        print(f'Binary Search till {result}')
        currentIter = 0
        while len(remaining_persons) > result:

            #For each iteration we augment the test image a little and recalculate the score but this time not with
            #entire dataset but half of it
            augmented_test_image = self.datagen.flow(np.expand_dims(img, axis=0), batch_size=1)
            # Get embeddings from the test image
            test_embedding = self.model.get_embeddings(augmented_test_image).numpy()     
            num_iterations += 1
            round_scores = []
            for person in remaining_persons:
                similarities = self.calculate_similarity(test_embedding, all_embeddings[person])
                round_scores.append((person, max(similarities)))  # Choose the max similarity in this round
                print(f"Round : {currentIter} Person : {person} Score: {max(similarities)}")
            currentIter+=1
            
            print(f"Round {num_iterations} scores: {round_scores}")
            # Sort and keep top N/2
            round_scores.sort(key=lambda x: x[1], reverse=True)
            remaining_persons = [person for person, _ in round_scores[:len(round_scores) // 2]]

        # Final round: compare all embeddings of top 10
        frequency_count = {person: 0 for person in remaining_persons}
        average_scores = {person: 0 for person in remaining_persons}

        for person in remaining_persons:
            similarities = self.calculate_similarity(test_embedding, all_embeddings[person])
            frequency_count[person] += sum(1 for sim in similarities if sim > 0.8)  # Example threshold
            average_scores[person] = np.mean(similarities)

        # Debugging: Print final scores
        print(f"Frequency count: {frequency_count}")
        print(f"Average scores: {average_scores}")

        # Rank by both average score and frequency count
        prediction_by_average = sorted(average_scores.items(), key=lambda x: x[1], reverse=True)
        prediction_by_frequency = sorted(frequency_count.items(), key=lambda x: x[1], reverse=True)

        return prediction_by_average, prediction_by_frequency

    def evaluate(self):
        """
        Evaluate the model using the test set and embeddings.
        """
        correct_predictions_avg = 0
        correct_predictions_freq = 0
        total_tests = 0

        for folder_name in os.listdir(self.test_dir):
            test_folder = os.path.join(self.test_dir, folder_name)
            if os.path.isdir(test_folder):
                for image_name in os.listdir(test_folder):
                    test_image_path = os.path.join(test_folder, image_name)
                    prediction_avg, prediction_freq = self.predict(test_image_path)

                    # Check if the top prediction matches the true label
                    total_tests += 1
                    if not prediction_freq or not prediction_avg[0] or not prediction_avg[0][0]:
                        print(f"No predictions averages for {folder_name}. Skipping.")
                    elif prediction_avg[0][0] == folder_name:
                        correct_predictions_avg += 1
                    if not prediction_freq or not prediction_freq[0] or not prediction_freq[0][0]:
                        print(f"No predictions frequencies for {folder_name}. Skipping.")
                    elif prediction_freq[0][0] == folder_name:
                            correct_predictions_freq += 1

        accuracy_avg = correct_predictions_avg / total_tests * 100
        accuracy_freq = correct_predictions_freq / total_tests * 100

        print(f"Accuracy using average score: {accuracy_avg:.2f}%")
        print(f"Accuracy using frequency score: {accuracy_freq:.2f}%")
