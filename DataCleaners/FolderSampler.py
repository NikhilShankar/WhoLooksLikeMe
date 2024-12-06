import os
import random
import shutil

class FolderSampler:
    def __init__(self, source_dir, destination_dir):
        """
        Initialize the FolderSampler with source and destination directories.
        """
        self.source_dir = source_dir
        self.destination_dir = destination_dir

        # Create the destination directory if it doesn't exist
        if not os.path.exists(self.destination_dir):
            os.makedirs(self.destination_dir)

    def copy_random_folders(self, num_folders=50):
        """
        Randomly selects a given number of folders from the source directory
        and copies them to the destination directory.

        :param num_folders: Number of folders to copy (default is 50)
        """
        # Get the list of all folders in the source directory
        all_folders = [folder for folder in os.listdir(self.source_dir)
                       if os.path.isdir(os.path.join(self.source_dir, folder))]

        # Ensure the source has enough folders
        if len(all_folders) < num_folders:
            print(f"Error: The source directory has only {len(all_folders)} folders, "
                  f"but you requested {num_folders}.")
            return

        # Randomly select the specified number of folders
        selected_folders = random.sample(all_folders, num_folders)

        # Copy the selected folders to the destination directory
        for folder in selected_folders:
            src_folder = os.path.join(self.source_dir, folder)
            dst_folder = os.path.join(self.destination_dir, folder)

            # Copy the folder and its contents
            shutil.copytree(src_folder, dst_folder)
            print(f"Copied {folder} to {self.destination_dir}")

        print(f"Successfully copied {num_folders} random folders to {self.destination_dir}")

