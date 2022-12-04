"""
This program processes all images in the dataset to make them suitable for training and testing

Only processes .png files in the dataset which are not names base renaming them with -out appended
"""

import os
import image_processor
from pathlib import Path

DATASET_FOLDER_NAME = "JigsawDatasetOutput"


def main():
    dataset_location = Path().absolute().joinpath(DATASET_FOLDER_NAME)
    process_dataset(dataset_location)


def process_dataset(dataset_location):
    for folder_name in os.listdir(dataset_location):
        folder_path = os.path.join(dataset_location, folder_name)
        if os.path.isdir(folder_path):
            process_image_folder(folder_path)


def process_image_folder(folder_path):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        # Validate if training image not base or labels
        if not file_path.endswith(".png"):
            continue
        if file_path.endswith("base.png"):
            continue

        process_image(file_path)


def process_image(file_path):
    image_processor.process_from_code('"' + file_path + '"' + " --quiet")


if __name__ == "__main__":
    main()
