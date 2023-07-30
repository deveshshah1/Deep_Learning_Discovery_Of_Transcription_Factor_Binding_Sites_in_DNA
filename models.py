"""
Author: Devesh Shah
Project Title: Deep Learning Discovery Of Transcription Factor Binding Sites in DNA

This file represents the various networks we will use for training. We implement several simplistic models here
for reference. The key idea is to study the impact of different network types. We are not currently investigating
the best design for each model type.

The implemented model types include:
    - Convolutional Network (TODO: CHANGE TO SIGMOID)
    - Multilayer Perceptron
"""

import torch.nn as nn
import torch.nn.functional as F


class ConvNetwork(nn.Module):
    """
    The ConvNetwork leverages 1D convolutional layers to traverse the input DNA sequence. It can be run using
    either 1 or 2 conv layers followed by 2 fc layers. The output is provided as a softmax(x) to represent the
    probability of each class.

    Note: Unlike computer vision applications, we typically see better performance on DNA sequencing using a smaller
    number of layers.
    """
    def __init__(self, num_conv_layers=1, input_size=50, k1=12, k2=4):
        """
        :param num_conv_layers: Either 1 or 2 indicating the number of convolutional layers before the linear layers
        :param input_size: The input size of the DNA sequence. Set to 50 for our dataset
        :param k1: The kernel size of the first conv layer
        :param k2: The kernel size of the second conv layer. Only used if num_conv_layers=2
        """
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=k1)

        # Create second conv layer if init function parameter is set to 2.
        self.second_conv = True if num_conv_layers==2 else False
        if self.second_conv:
            self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=k2)
            self.lin_len = ((input_size-k1-k2+2)//4)*32
            self.fc1 = nn.Linear(in_features=self.lin_len, out_features=16)
        else:
            self.lin_len = ((input_size-k1+1)//4) * 32
            self.fc1 = nn.Linear(in_features=self.lin_len, out_features=16)
        self.out = nn.Linear(in_features=16, out_features=2)

    def forward(self, x):
        """
        :param x: Our input tensor of shape (batch, num_channels, input_size)
        :return: Our model predictions for each input of shape (batch, 2)
        """
        x = F.relu(self.conv1(x))

        # Only run through second conv layer if was set during initialization of model
        if self.second_conv:
            x = F.relu(self.conv2(x))

        x = F.max_pool1d(x, kernel_size=4)
        x = x.reshape(-1, self.lin_len)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.out(x))
        return x
