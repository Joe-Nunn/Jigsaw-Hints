"""
Contains class NeuralNetwork which defines a neural network to predict image labels for actions in sports
"""

import torch
import torch.nn as nn
import conv_image_processor


class NeuralNetwork(nn.Module):
    """
    Neural network to predict probabilities of a jigsaw piece being from the same part of the puzzle as a section
    of the base image
    """
    def __init__(self):
        super().__init__()
        # Comparing two flattened tensors representing the jigsaw piece and section of the board
        self.fc1 = nn.Linear(conv_image_processor.FLATTENED_TENSOR_SIZE * 2, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 1)
        self.dropout = nn.Dropout()

        def forward(self, batch):
            """
            Calculates prediction that jigsaw piece and the sample of the base are the same location
            """
            # Fully connected layers
            batch = self.fc1(batch)
            batch = self.dropout(batch)
            batch = self.fc2(batch)
            batch = self.fc3(batch)
            batch = torch.sigmoid(batch)
            return batch
