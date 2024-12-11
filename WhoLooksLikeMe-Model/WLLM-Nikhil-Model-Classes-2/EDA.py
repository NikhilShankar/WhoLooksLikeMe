import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image
from collections import Counter

class EDA:
    def __init__(self, image_dir, output_dir):
        """
        Initializes the EDA class with the image directory and output directory for saving plots and data.
        
        :param image_dir: Path to the root directory containing personality folders with images.
        :param output_dir: Path to the directory where plots and data will be saved.
        """
        self.image_dir = image_dir
        self.output_dir = output_dir
        
        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
    def calculate_eda(self):
        """
        Calculate and plot all requested EDA steps (bar graph, histograms, image type counts, and display random images).
        """
        # Step 1: Get image file paths and metadata
        image_data = self._get_image_data()
        
        # Step 2: Plot bar graph for number of images per personality
        self._plot_image_count_bargraph(image_data)
        
        # Step 3: Plot histogram of image widths
        self._plot_image_width_histogram(image_data)
        
        # Step 4: Plot histogram of image heights
        self._plot_image_height_histogram(image_data)
        
        # Step 5: Count image types (jpg, png, jpeg, etc.)
        self._plot_image_type_distribution(image_data)
        
        # Step 6: Display a random selection of 50 images in a grid
        self._display_random_images(image_data)
        
        # Step 7: Save the data (DataFrames)
        self._save_dataframes(image_data)

    def _get_image_data(self):
        """
        Helper function to retrieve all image file paths and their metadata (width, height, file type).
        
        :return: A DataFrame containing image information
        """
        image_data = []
        image_types = []
        for root, dirs, files in os.walk(self.image_dir):
            for file in files:
                # Only consider image files
                if file.lower().endswith(('jpg', 'jpeg', 'png')):
                    # Get the personality name (parent folder)
                    personality = os.path.basename(root)
                    # Full path to the image
                    image_path = os.path.join(root, file)
                    
                    # Open image to get width and height
                    with Image.open(image_path) as img:
                        width, height = img.size
                        image_types.append(file.split('.')[-1].lower())
                    
                    image_data.append({
                        'Personality': personality,
                        'Image Path': image_path,
                        'Width': width,
                        'Height': height,
                        'File Type': file.split('.')[-1].lower()
                    })
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(image_data)
        return df, image_types

    def _plot_image_count_bargraph(self, image_data):
        """
        Plot and save the bar graph for the number of images for each personality.
        """
        df, _ = image_data
        image_counts = df['Personality'].value_counts()
        
        # Ensure output subdirectory exists for plots
        plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        
        # Plot the bar graph
        plt.figure(figsize=(10, 6))
        sns.barplot(x=image_counts.index, y=image_counts.values, palette='viridis')
        plt.title('Number of Images per Personality')
        plt.xlabel('Personality')
        plt.ylabel('Number of Images')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()  # Display the plot
        plt.savefig(os.path.join(plot_dir, 'image_count_bargraph.png'))
        plt.close()

        # Display dataframe with max and min counts
        min_personality = image_counts.idxmin()
        max_personality = image_counts.idxmax()
        min_count = image_counts.min()
        max_count = image_counts.max()

        print(f"Personality with minimum images: {min_personality} ({min_count} images)")
        print(f"Personality with maximum images: {max_personality} ({max_count} images)")

        # Save the counts as a DataFrame
        image_counts_df = image_counts.reset_index()
        image_counts_df.columns = ['Personality', 'Image Count']
        image_counts_df.to_csv(os.path.join(self.output_dir, 'image_count_per_personality.csv'), index=False)

    def _plot_image_width_histogram(self, image_data):
        """
        Plot and save the histogram of image widths.
        """
        df, _ = image_data
        
        # Ensure output subdirectory exists for plots
        plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        plt.figure(figsize=(10, 6))
        sns.histplot(df['Width'], kde=True, bins=20, color='skyblue')
        plt.title('Distribution of Image Widths')
        plt.xlabel('Width (pixels)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()  # Display the plot
        plt.savefig(os.path.join(plot_dir, 'image_width_histogram.png'))
        plt.close()

    def _plot_image_height_histogram(self, image_data):
        """
        Plot and save the histogram of image heights.
        """
        df, _ = image_data
        
        # Ensure output subdirectory exists for plots
        plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        plt.figure(figsize=(10, 6))
        sns.histplot(df['Height'], kde=True, bins=20, color='salmon')
        plt.title('Distribution of Image Heights')
        plt.xlabel('Height (pixels)')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()  # Display the plot
        plt.savefig(os.path.join(plot_dir, 'image_height_histogram.png'))
        plt.close()

    def _plot_image_type_distribution(self, image_data):
        """
        Plot and save the distribution of image file types (jpg, png, jpeg, etc.).
        """
        _, image_types = image_data
        type_counts = Counter(image_types)

        # Ensure output subdirectory exists for plots
        plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        # Plot the distribution of image types
        plt.figure(figsize=(8, 6))
        sns.barplot(x=list(type_counts.keys()), y=list(type_counts.values()), palette='Set2')
        plt.title('Distribution of Image File Types')
        plt.xlabel('File Type')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()  # Display the plot
        plt.savefig(os.path.join(plot_dir, 'image_type_distribution.png'))
        plt.close()

        # Save the counts as a DataFrame
        type_counts_df = pd.DataFrame(list(type_counts.items()), columns=['File Type', 'Count'])
        type_counts_df.to_csv(os.path.join(self.output_dir, 'image_file_type_counts.csv'), index=False)

    def _display_random_images(self, image_data):
        """
        Randomly select 50 images and display them in a 10x5 grid.
        """
        df, _ = image_data
        
        # Randomly select 50 images
        random_images = random.sample(df['Image Path'].tolist(), 50)
        
        # Set up the grid (10 rows, 5 columns)
        fig, axes = plt.subplots(10, 5, figsize=(15, 20))
        axes = axes.ravel()
        
        # Loop through the axes and display the images
        for i in range(50):
            img_path = random_images[i]
            img = Image.open(img_path)
            axes[i].imshow(img)
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()  # Display the plot
        plt.savefig(os.path.join(self.output_dir, 'random_images_grid.png'))
        plt.close()

    def _save_dataframes(self, image_data):
        """
        Save the DataFrames as CSV files.
        """
        df, _ = image_data
        
        # Ensure output directory exists for data
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Save the full image data
        df.to_csv(os.path.join(self.output_dir, 'image_metadata.csv'), index=False)
        print("All EDA results have been saved in the output directory.")


# Example usage:
# eda = EDA(image_dir='/path/to/images', output_dir='/path/to/save/plots_and_data')
# eda.calculate_eda()
