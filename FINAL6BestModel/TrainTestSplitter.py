import os
import random
import shutil

class TrainTestSplitter:
    def __init__(self, dataset_dir):
        """
        Initialize the DatasetSplitter with the dataset folder path and the number of files for the test split.
        This will create 'train' and 'test' directories inside the dataset directory.
        """
        self.dataset_dir = dataset_dir
        self.train_dir = os.path.join(dataset_dir, 'train')
        self.test_dir = os.path.join(dataset_dir, 'test')

        # Create train and test directories if they don't exist
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)

    def create_train_test_split(self, test_file_count=1):
        """
        Split the dataset into train and test folders. 
        n random images will go into the test folder, and the rest will go into the train folder.
        """
        # Loop over each folder (personality) in the dataset directory
        for folder_name in os.listdir(self.dataset_dir):
            folder_path = os.path.join(self.dataset_dir, folder_name)

            # Skip the train/test folders to avoid re-processing
            if folder_name in ['train', 'test']:
                continue

            if os.path.isdir(folder_path):
                # Create corresponding folder inside train and test directories
                train_folder_path = os.path.join(self.train_dir, folder_name)
                test_folder_path = os.path.join(self.test_dir, folder_name)

                if not os.path.exists(train_folder_path):
                    os.makedirs(train_folder_path)
                if not os.path.exists(test_folder_path):
                    os.makedirs(test_folder_path)

                # Get all image files in the folder
                image_files = [f for f in os.listdir(folder_path) if f.endswith(('jpg', 'jpeg', 'png'))]

                # Ensure n does not exceed the number of available images
                test_file_count = min(test_file_count, len(image_files))

                # Randomly select n images for the test folder
                test_images = random.sample(image_files, test_file_count)

                # Move the selected test images to the test folder
                for test_image in test_images:
                    test_image_path = os.path.join(folder_path, test_image)
                    shutil.move(test_image_path, os.path.join(test_folder_path, test_image))

                # Move the remaining images to the train folder
                for image_file in image_files:
                    if image_file not in test_images:  # Skip the selected test images
                        image_path = os.path.join(folder_path, image_file)
                        shutil.move(image_path, os.path.join(train_folder_path, image_file))

                print(f"Moved {test_file_count} images to test and the rest to train for {folder_name}.")
