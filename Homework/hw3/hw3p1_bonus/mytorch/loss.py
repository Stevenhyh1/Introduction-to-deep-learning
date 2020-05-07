# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os

# Ok now things get decidedly more interesting. The following Criterion class
# will be used again as the basis for a number of loss functions (which are in the
# form of classes so that they can be exchanged easily (it's how PyTorch and other
# ML libraries do it))

class Criterion(object):
    """
    Interface for loss functions.
    """

    # Nothing needs done to this class, it's used by the following Criterion classes

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplementedError

    def derivative(self):
        raise NotImplementedError

class SoftmaxCrossEntropy(Criterion):
    """
    Softmax loss
    """

    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()

    def forward(self, x, y):
        """
        Argument: 
            x (np.array): (batch size, 10)
            y (np.array): (batch size, 10)
        Return:
            out (np.array): (batch size, )
        """
        self.logits = x
        self.labels = y

        C = np.max(x, axis = 1, keepdims = True)
        expsum = np.sum(np.exp(x - C), axis = 1, keepdims = True)
        logsum = C + np.log(expsum)
        self.sm = np.exp(x - logsum)
        out = np.sum(-y * np.log(self.sm), axis = 1)

        return out

    def derivative(self):
        """
        Return:
            out (np.array): (batch size, 10)
        """

        return self.sm - self.labels