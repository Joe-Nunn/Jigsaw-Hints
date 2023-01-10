import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from neural_network import NeuralNetwork
from jigsaw_piece_dataset import JigsawPieceDataset

DEFAULT_TRAINING_SIZE = 0.8  # Default proportion of dataset to use for training compared to testing
DEFAULT_BATCH_SIZE = 128


class Trainer:
    """
    Trains and tests a neural network to identify if an image of a jigsaw piece comes from a section of the base image
    """

    def __init__(self, dataset_path, csv_name, batch_size=DEFAULT_BATCH_SIZE, training_size=DEFAULT_TRAINING_SIZE):
        self.network = NeuralNetwork
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

        self.batch_size = batch_size
        self.training_size = training_size
        self.train_loader, self.test_loader = self.__prepare_dataset(dataset_path, csv_name)

    def __prepare_dataset(self, folder_path, csv_name):
        """
        Loads the dataset splitting it into training and testing sets
        Returns data loaders for the training and testing set
        """
        dataset = JigsawPieceDataset(csv_name, folder_path)

        # Calculate number of images to be used for training and testing
        n_train_images = round(dataset.__len__() * self.training_size)
        n_test_images = dataset.__len__() - n_train_images

        # Split the dataset into training and testing set
        train_set, test_set = torch.utils.data.random_split(dataset, [n_train_images, n_test_images])

        # Create data loaders from split dataset
        train_loader = DataLoader(dataset=train_set, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_set, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader
