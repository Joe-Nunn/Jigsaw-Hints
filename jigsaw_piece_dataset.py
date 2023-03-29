import random
import torch
import os
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

INPUT_IMAGE_SIZE = 256  # x and y


class JigsawPieceDataset(Dataset):
    """
    Loads and manages the dataset of jigsaw pieces
    """

    def __init__(self, csv_file_name, directory_path):
        self.root_dir = directory_path
        csv_path = os.path.join(self.root_dir, csv_file_name)
        self.annotations = pd.read_csv(csv_path)
        self.__split_dataset_correct_cords()

    def __split_dataset_correct_cords(self):
        """
        Modifies dataset giving half the pieces fake coordinates of where the piece originates in the base image
        Adds another column to the data frame indicating if the piece coordinates are wrong
        """
        matching = []
        is_matching = True

        for i in range(self.__len__()):
            matching.append(is_matching)
            if not is_matching:  # modify row to give wrong coordinates
                # Get max y and x using base image size
                base_path = os.path.join(self.root_dir, self.annotations.at[i, "base_path"])
                base = Image.open(base_path)
                size_x, size_y = base.size

                # Generate lower x and y corner where the top and right coords will be within the bounds of the image
                new_lower_left_x = int(random.random() * (size_x - INPUT_IMAGE_SIZE))
                new_lower_left_y = int(random.random() * (size_y - INPUT_IMAGE_SIZE))

                # Set all the new x coords
                self.annotations.at[i, "lower_left_x"] = new_lower_left_x / size_x
                self.annotations.at[i, "top_left_x"] = new_lower_left_x / size_x
                self.annotations.at[i, "top_right_x"] = (new_lower_left_x + INPUT_IMAGE_SIZE - 1) / size_x
                self.annotations.at[i, "bottom_right_x"] = (new_lower_left_x + INPUT_IMAGE_SIZE - 1) / size_x

                # Set all the new y coords
                self.annotations.at[i, "lower_left_y"] = new_lower_left_y / size_y
                self.annotations.at[i, "top_left_y"] = (new_lower_left_y + INPUT_IMAGE_SIZE - 1) / size_y
                self.annotations.at[i, "top_right_y"] = (new_lower_left_y + INPUT_IMAGE_SIZE - 1) / size_y
                self.annotations.at[i, "lower_left_y"] = new_lower_left_y / size_y

            is_matching = not is_matching

        # Add column indicating which origin coords for the piece are correct
        self.annotations['correct_base_cords'] = matching

    def __len__(self):
        """
        Get size of dataset
        """
        return len(self.annotations)

    def __getitem__(self, index):
        """
        Returns a tuple of:
            A tensor of the jigsaw piece,
            a tensor of a section of the base,
            and a tensor of 0 or 1 indicating whether the jigsaw piece is actually from the base section image
        """
        piece_id = self.annotations.at[index, "piece_id"]
        piece_path = os.path.join(self.root_dir, piece_id + ".png")
        piece = Image.open(piece_path)
        piece = piece.convert("RGB")  # remove alpha channel
        piece.rotate(90 * random.randint(0, 3))  # Random rotation or no rotation
        transform = transforms.ToTensor()
        piece = transform(piece)
        piece = piece.float()

        base_path = os.path.join(self.root_dir, self.annotations.at[index, "base_path"])
        base = Image.open(base_path)
        max_x, max_y = base.size
        left = int(self.annotations.at[index, "lower_left_x"] * max_x)
        right = left + INPUT_IMAGE_SIZE
        # The y in annotations is from the bottom while in the image from top so have to the flip y from bottom to top
        bottom = max_y - int(self.annotations.at[index, "lower_left_y"] * max_y)
        top = bottom - INPUT_IMAGE_SIZE
        base_section = base.crop((left, top, right, bottom))
        base_section = transform(base_section)
        base_section = base_section.float()

        correct_base_cords = self.annotations.at[index, "correct_base_cords"]
        correct_base_cords = torch.FloatTensor([correct_base_cords])

        return piece, base_section, correct_base_cords
