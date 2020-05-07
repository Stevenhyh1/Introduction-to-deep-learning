# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os
import sys

sys.path.append('mytorch')
from loss import *
from activation import *
from linear import *
from conv import *


class CNN_SimpleScanningMLP():
    def __init__(self):
        ## Your code goes here -->
        # self.conv1 = ???
        # self.conv2 = ???
        # self.conv3 = ???
        # ...
        # <---------------------
        self.conv1 = Conv1D(24, 8, 8, 4)
        self.conv2 = Conv1D(8, 16, 1, 1)
        self.conv3 = Conv1D(16, 4, 1, 1)
        self.layers = []
        self.layers.append(self.conv1)
        self.layers.append(ReLU())
        self.layers.append(self.conv2)
        self.layers.append(ReLU())
        self.layers.append(self.conv3)
        self.layers.append(Flatten())

    def __call__(self, x):
        # Do not modify this method
        return self.forward(x)

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN

        w1,w2,w3 = weights
        # import pdb; pdb.set_trace()
        w1 = w1.T.reshape(8, 8, 24)
        w1 = np.transpose(w1, (0,2,1))
        w2 = w2.T.reshape(16, 1, 8)
        w2 = np.transpose(w2, (0,2,1))
        w3 = w3.T.reshape(4, 1, 16)
        w3 = np.transpose(w3, (0,2,1))
        self.conv1.W = w1
        self.conv2.W = w2
        self.conv3.W = w3

    def forward(self, x):
        """
        Do not modify this method

        Argument:
            x (np.array): (batch size, in channel, in width)
        Return:
            out (np.array): (batch size, out channel , out width)
        """

        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, delta):
        """
        Do not modify this method

        Argument:
            delta (np.array): (batch size, out channel, out width)
        Return:
            dx (np.array): (batch size, in channel, in width)
        """

        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta


class CNN_DistributedScanningMLP():
    def __init__(self):
        ## Your code goes here -->
        # self.conv1 = ???
        # self.conv2 = ???
        # self.conv3 = ???
        # ...
        # <---------------------
        self.conv1 = Conv1D(24, 2, 2, 2)
        self.conv2 = Conv1D(2, 8, 2, 2)
        self.conv3 = Conv1D(8, 4, 2, 1)
        self.layers = []
        self.layers.append(self.conv1)
        self.layers.append(ReLU())
        self.layers.append(self.conv2)
        self.layers.append(ReLU())
        self.layers.append(self.conv3)
        self.layers.append(Flatten())

    def __call__(self, x):
        # Do not modify this method
        return self.forward(x)

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN

        w1, w2, w3 = weights
        # import pdb; pdb.set_trace()
        w1 = w1[:48,:2]
        w1 = w1.T.reshape(2, 2, 24)
        w1 = np.transpose(w1, (0,2,1))

        w2 = w2[:4,:8]
        w2 = w2.T.reshape(8, 2, 2)
        w2 = np.transpose(w2, (0,2,1))

        w3 = w3.T.reshape(4, 2, 8)
        w3 = np.transpose(w3, (0,2,1))
        self.conv1.W = w1
        self.conv2.W = w2
        self.conv3.W = w3

    def forward(self, x):
        """
        Do not modify this method

        Argument:
            x (np.array): (batch size, in channel, in width)
        Return:
            out (np.array): (batch size, out channel , out width)
        """

        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, delta):
        """
        Do not modify this method

        Argument:
            delta (np.array): (batch size, out channel, out width)
        Return:
            dx (np.array): (batch size, in channel, in width)
        """

        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta
