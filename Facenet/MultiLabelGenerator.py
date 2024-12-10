import os
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

class MultiLabelGenerator(Sequence):
    def __init__(self, directory, batch_size, target_size=(299, 299), augmentations=None):
        """
        Data generator where the label for each image is its folder name.
        
        Args:
            directory (str): Path to the directory containing class subfolders.
            batch_size (int): Number of samples per batch.
            target_size (tuple): Tuple of integers (height, width) for resizing images.
            augmentations (ImageDataGenerator): Instance of ImageDataGenerator for augmentation.
        """
        self.directory = directory
        self.batch_size = batch_size
        self.target_size = target_size
        self.augmentations = augmentations or ImageDataGenerator(rescale=1.0 / 255)

        # Prepare image paths and corresponding folder names as labels
        self.image_paths = []
        self.labels = []

        for class_folder in sorted(os.listdir(directory)):
            class_dir = os.path.join(directory, class_folder)
            if os.path.isdir(class_dir):  # Only process directories
                for filename in os.listdir(class_dir):
                    full_path = os.path.join(class_dir, filename)
                    self.image_paths.append(full_path)
                    self.labels.append(class_folder)  # Use folder name as label
        
        self.num_samples = len(self.image_paths)
        self.label_to_index = {label: idx for idx, label in enumerate(sorted(set(self.labels)))}
        self.labels = np.array([self.label_to_index[label] for label in self.labels])  # Convert to numeric labels

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, self.num_samples)
        batch_paths = self.image_paths[start:end]
        batch_labels = self.labels[start:end]

        # Load and augment images
        images = []
        for path in batch_paths:
            img = load_img(path, target_size=self.target_size)
            img_array = img_to_array(img)
            augmented_img = self.augmentations.random_transform(img_array)  # Apply augmentations
            images.append(augmented_img)
        
        batch_primary_labels = np.array(batch_labels, dtype=np.int32)
        
        images = np.array(images) / 255.0  # Normalize images
        return (images, (batch_primary_labels))

    def on_epoch_end(self):
        indices = np.arange(self.num_samples)
        np.random.shuffle(indices)
        self.image_paths = [self.image_paths[i] for i in indices]
        self.labels = self.labels[indices]