"""
Contains class NeuralNetwork which defines a neural network to predict image labels for actions in sports
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

FLATTENED_TENSOR_SIZE = 262144


class NeuralNetwork(nn.Module):
    """
    Neural network to predict probabilities of a jigsaw piece being from the same part of the puzzle as a section
    of the base image
    """
    def __init__(self):
        super().__init__()
        # Half size, will be applied after every conv layer. E.g. 128x128 becomes 64x64
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # maintains size and ratio. E.g. input of 256x256 stays 256x256
        self.conv_piece_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_piece_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_piece_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.conv_base_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv_base_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv_base_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        # Comparing two flattened tensors representing the jigsaw piece and section of the board
        self.fc1 = nn.Linear(FLATTENED_TENSOR_SIZE, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 1)
        self.dropout = nn.Dropout()

    def forward(self, pieces, base_sections):
        """
        Calculates prediction that jigsaw piece and the sample of the base are the same location
        """
        pieces = self.conv_piece_1(pieces)
        pieces = F.relu(pieces)
        pieces = self.pool(pieces)
        pieces = self.conv_piece_2(pieces)
        pieces = F.relu(pieces)
        pieces = self.pool(pieces)
        pieces = self.conv_piece_3(pieces)
        pieces = F.relu(pieces)
        pieces = self.pool(pieces)

        base_sections = self.conv_base_1(base_sections)
        base_sections = F.relu(base_sections)
        base_sections = self.pool(base_sections)
        base_sections = self.conv_base_2(base_sections)
        base_sections = F.relu(base_sections)
        base_sections = self.pool(base_sections)
        base_sections = self.conv_base_3(base_sections)
        base_sections = F.relu(base_sections)
        base_sections = self.pool(base_sections)

        #  Dimension of 1 is after batch dimension
        combined_batch = torch.concat((pieces, base_sections), dim=1)
        combined_batch = torch.flatten(combined_batch, start_dim=1)

        combined_batch = self.fc1(combined_batch)
        combined_batch = self.dropout(combined_batch)
        combined_batch = self.fc2(combined_batch)
        combined_batch = self.fc3(combined_batch)
        combined_batch = torch.sigmoid(combined_batch)
        return combined_batch
