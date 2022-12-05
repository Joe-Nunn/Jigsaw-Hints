"""
Contains class NeuralNetwork which defines a neural network to predict image labels for actions in sports
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    """
    Neural network to predict probabilities of a jigsaw piece being from the same part of the puzzle as a section
    of the base image
    """
    def __init__(self):
        super().__init__()

        # Half size, will be applied after every conv layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # maintains size of 256x512
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        # maintains size of 128x256
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Original image of size 256x512 is halved twice to 64 * 128 with 64 channels
        self.fc1 = nn.Linear(64 * 128 * 64, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 1)
        self.dropout = nn.Dropout()

        def forward(self, batch):
            """
            Calculates prediction of if the two images combined - the jigsaw piece and the sample of the base are the
            same location
            """

            # Convolutional layers
            batch = self.conv1(batch)
            batch = F.relu(batch)
            batch = self.pool(batch)
            batch = self.conv2(batch)
            batch = F.relu(batch)
            batch = self.pool(batch)

            # Flatten
            batch = batch.reshape(batch.shape[0], -1)

            # Fully connected layers
            batch = self.fc1(batch)
            batch = self.dropout(batch)
            batch = self.fc2(batch)
            batch = self.fc3(batch)
            batch = torch.sigmoid(batch)
            return batch
