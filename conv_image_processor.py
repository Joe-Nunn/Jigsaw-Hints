"""
Class to create flattened feature maps of jigsaw pieces and sections of the overall board
"""

import torch.nn as nn
import torch.nn.functional as F

LAYERS = 3

INPUT_IMAGE_SIZE = 256  # Height and width same

FINAL_OUT_CHANNELS = 128

FLATTENED_TENSOR_SIZE = (INPUT_IMAGE_SIZE / (pow(pow(2, LAYERS), 2))) * FINAL_OUT_CHANNELS


class ConvImageProcessor(nn.Module):
    """
    Processes images to create flattened feature maps of jigsaw pieces and sections of the overall board
    """

    def __init__(self):
        super().__init__()

        # Half size, will be applied after every conv layer. E.g. 128x128 becomes 64x64
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # maintains size and ratio. E.g. input of 256x256 stays 256x256
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=FINAL_OUT_CHANNELS, kernel_size=3, stride=1, padding=1)

    def forward(self, image_tensor):
        """
        Creates feature map of an image and returns it flattened
        """
        # Convolutional layers
        image_tensor = self.conv1(image_tensor)
        image_tensor = F.relu(image_tensor)
        image_tensor = self.pool(image_tensor)
        image_tensor = self.conv2(image_tensor)
        image_tensor = F.relu(image_tensor)
        image_tensor = self.pool(image_tensor)
        image_tensor = self.conv3(image_tensor)
        image_tensor = self.relu(image_tensor)
        image_tensor = self.pool(image_tensor)

        # Flatten
        image_tensor = image_tensor.reshape(image_tensor.shape[0], -1)

        return image_tensor
