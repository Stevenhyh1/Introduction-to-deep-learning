import numpy as np
from activation import *

class RNN_Cell(object):
    def __init__(self, input_size, hidden_size):

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Activation function for
        self.activation = Tanh()

        # hidden dimension and input dimension
        h = self.hidden_size
        d = self.input_size

        # Weights and biases
        self.W_ih = np.random.randn(h, d)
        self.W_hh = np.random.randn(h, h)
        self.b_ih = np.random.randn(h)
        self.b_hh = np.random.randn(h)

        # Gradients
        self.dW_ih = np.zeros((h, d))
        self.dW_hh = np.zeros((h, h))

        self.db_ih = np.zeros(h)
        self.db_hh = np.zeros(h)

    def init_weights(self, W_ih, W_hh, b_ih, b_hh):
        self.W_ih = W_ih
        self.W_hh = W_hh
        self.b_ih = b_ih
        self.b_hh = b_hh

    def zero_grad(self):
        d = self.input_size
        h = self.hidden_size
        self.dW_ih = np.zeros((h, d))
        self.dW_hh = np.zeros((h, h))
        self.db_ih = np.zeros(h)
        self.db_hh = np.zeros(h)

    def __call__(self, x, h):
        return self.forward(x, h)

    def forward(self, x, h):
        """
        RNN cell forward (single time step)

        Input
        ----------
        x : (batch_size, input_size)
            Input
        h : (batch_size, hidden_size)
            Hidden states of previous time step: H(t-1)

        Returns
        -------
        h : (batch_size, hidden_size)
            New hidden states of the current time step
        """

        h_prime = self.activation(np.dot(x, self.W_ih.T) + self.b_ih + np.dot(h, self.W_hh.T) + self.b_hh)

        return h_prime

    def backward(self, delta, h, h_prev_l, h_prev_t):
        """
        RNN cell backward (single time step)

        Parameters (see writeup for more details)
        ----------
        delta : (batch_size, hidden_size)
            Gradient w.r.t the current hidden layer H(t, l)
            = (gradient from the next layer at current time step + the gradient from current layer at next time step)
        h : (batch_size, hidden_size)
            Hidden states of current time step, current layer H(t, l)
        h_prev_l: (batch_size, input_size)
            Hidden states of current time step, previous layer H(t, l-1), or input X(t) if this is the first layer
        h_prev_t: (batch_size, hidden_size)
            Hidden states of previous time step, current layer H(t-1, l)

        Returns
        -------
        dx : (batch_size, input_size)
            Derivative w.r.t. the previous layer, current time step
        dh : (batch_size, hidden_size)
            Derivative w.r.t. the current layer, previous time step
        """

        batch_size = delta.shape[0]
        dz = self.activation.derivative(state=h) * delta

        # Add Your Code Here -->
        # Compute the gradients of the weights, biases, dx and dh
        self.dW_ih += np.dot(dz.T, h_prev_l) / batch_size
        self.dW_hh += np.dot(dz.T, h_prev_t) / batch_size
        self.db_ih += dz.sum(axis=0) / batch_size
        self.db_hh += dz.sum(axis=0) / batch_size
        dx = np.dot(dz, self.W_ih)
        dh = np.dot(dz, self.W_hh)
        # <-------------------

        return dx, dh
