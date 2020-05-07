# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os


class Activation(object):

    """
    Interface for activation functions (non-linearities).

    In all implementations, the state attribute must contain the result,
    i.e. the output of forward (it will be tested).
    """

    # No additional work is needed for this class, as it acts like an
    # abstract base class for the others

    # Note that these activation functions are scalar operations. I.e, they
    # shouldn't change the shape of the input.

    def __init__(self):
        self.state = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class Identity(Activation):

    """
    Identity function (already implemented).
    """

    # This class is a gimme as it is already implemented for you as an example

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        self.state = x
        return x

    def derivative(self):
        return 1.0


class Sigmoid(Activation):

    """
    Sigmoid non-linearity
    """

    # Remember do not change the function signatures as those are needed
    # to stay the same for AutoLab.

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        # Might we need to store something before returning?
        self.state = 1/(1+np.exp(np.negative(x)))
        return self.state
        # raise NotImplemented

    def derivative(self):
        # Maybe something we need later in here...
        return self.state * (1-self.state)
        # raise NotImplemented


class Tanh(Activation):

    """
    Tanh non-linearity
    """

    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        # self.state = (np.exp(x) - np.exp(np.negative(x)))/(np.exp(x) + np.exp(np.negative(x)))
        self.state = np.tanh(x)
        return self.state
        # raise NotImplemented

    def derivative(self):
        return 1 - (self.state * self.state)
        # raise NotImplemented


class ReLU(Activation):

    """
    ReLU non-linearity
    """

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        self.state = np.copy(x)
        self.state[np.where(x<0)]=0
        return self.state
        # raise NotImplemented

    def derivative(self):
        # raise NotImplemented
        self.state[np.where(self.state>0)] = 1
        self.state[np.where(self.state<=0)] = 0
        return self.state
