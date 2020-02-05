"""
Follow the instructions provided in the writeup to completely
implement the class specifications for a basic MLP, optimizer, .
You will be able to test each section individually by submitting
to autolab after implementing what is required for that section
-- do not worry if some methods required are not implemented yet.

Notes:

The __call__ method is a special reserved method in
python that defines the behaviour of an object when it is
used as a function. For example, take the Linear activation
function whose implementation has been provided.

# >>> activation = Identity()
# >>> activation(3)
# 3
# >>> activation.forward(3)
# 3
"""

# DO NOT import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os
import sys

sys.path.append('mytorch')
from loss import *
from activation import *
from batchnorm import *
from linear import *


class MLP(object):

    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn,
                 bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0):

        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly

        # Initialize and add all your linear layers into the list 'self.linear_layers'
        # (HINT: self.foo = [ bar(???) for ?? in ? ])
        # (HINT: Can you use zip here?)
        layer = hiddens.copy()
        layer.insert(0,input_size)
        layer.append(output_size)
        self.linear_layers = [Linear(d1,d2,weight_init_fn,bias_init_fn) for d1,d2 in zip(layer[:-1],layer[1:])]

        # If batch norm, add batch norm layers into the list 'self.bn_layers'
        if self.bn:
            self.bn_layers = [BatchNorm(d1) for d1 in hiddens[:self.num_bn_layers]]


    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, input_size)
        Return:
            out (np.array): (batch size, output_size)
        """
        # Complete the forward pass through your entire MLP.
        for i in range(self.num_bn_layers):
            x = self.linear_layers[i](x)
            if self.train_mode:
                x = self.bn_layers[i](x)
            else:
                x = self.bn_layers[i](x, eval=True)
            x = self.activations[i](x)
        for j in range(len(self.linear_layers)-self.num_bn_layers):
            x = self.linear_layers[j+self.num_bn_layers](x)
            x = self.activations[j+self.num_bn_layers](x)
        self.output = x
        return self.output
        # raise NotImplemented

    def zero_grads(self):
        # Use numpyArray.fill(0.0) to zero out your backpropped derivatives in each
        # of your linear and batchnorm layers.
        for i in range(len(self.linear_layers)):
            layer = self.linear_layers[i]
            layer.dW.fill(0.0)
            layer.db.fill(0.0)
        for j in range(self.num_bn_layers):
            norm_layer = self.bn_layers[j]
            norm_layer.dgamma.fill(0.0)
            norm_layer.dbeta.fill(0.0)
        # raise NotImplemented

    def step(self):
        # Apply a step to the weights and biases of the linear layers.
        # Apply a step to the weights of the batchnorm layers.
        # (You will add momentum later in the assignment to the linear layers only
        # , not the batchnorm layers)

        for i in range(len(self.linear_layers)):
            # Update weights and biases here
            
            cur_layer = self.linear_layers[i]
            # without momentum
            # cur_layer.W -= self.lr*cur_layer.dW
            # cur_layer.b -= self.lr*cur_layer.db
            # with momentum
            cur_layer.momentum_W = self.momentum*cur_layer.momentum_W - self.lr*cur_layer.dW
            cur_layer.momentum_b = self.momentum*cur_layer.momentum_b - self.lr*cur_layer.db
            cur_layer.W += cur_layer.momentum_W
            cur_layer.b += cur_layer.momentum_b

        # Do the same for batchnorm layers
        for j in range(self.num_bn_layers):
            cur_norm = self.bn_layers[j]
            cur_norm.gamma -= self.lr*cur_norm.dgamma
            cur_norm.beta -= self.lr*cur_norm.dbeta
        # raise NotImplemented

    def backward(self, labels):
        # Backpropagate through the activation functions, batch norm and
        # linear layers.
        # Be aware of which return derivatives and which are pure backward passes
        # i.e. take in a loss w.r.t it's output.
        self.criterion(self.output,labels)
        delta = self.criterion.derivative()
        for j in range(len(self.linear_layers)-self.num_bn_layers):
            delta = self.activations[-j-1].derivative() * delta
            delta = self.linear_layers[-j-1].backward(delta)
        for i in range(self.num_bn_layers):
            delta = self.activations[self.num_bn_layers-i-1].derivative() * delta
            delta = self.bn_layers[self.num_bn_layers-i-1].backward(delta)
            delta = self.linear_layers[self.num_bn_layers-i-1].backward(delta)       
        # raise NotImplemented

    def error(self, labels):
        return (np.argmax(self.output, axis = 1) != np.argmax(labels, axis = 1)).sum()

    def total_loss(self, labels):
        return self.criterion(self.output, labels).sum()

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False

def get_training_stats(mlp, dset, nepochs, batch_size):

    train, val, _ = dset
    trainx, trainy = train
    valx, valy = val

    idxs = np.arange(len(trainx))

    training_losses = np.zeros(nepochs)
    training_errors = np.zeros(nepochs)
    validation_losses = np.zeros(nepochs)
    validation_errors = np.zeros(nepochs)

    # Setup ...

    for e in range(nepochs):
        print(f"Epoch: {e+1}")
        # Per epoch setup ...
        cur_training_loss = []
        cur_training_error = []
        cur_validation_loss = []
        cur_validation_errors = []


        for b in range(0, len(trainx), batch_size):
            
            # pass  # Remove this line when you start implementing this
            # Train ...
            
            mlp.train()
            mlp.zero_grads()
            mlp(trainx[b:b+batch_size])
            cur_training_error.append(mlp.error(trainy[b:b+batch_size])/batch_size)
            cur_training_loss.append(mlp.total_loss(trainy[b:b+batch_size])/batch_size)
            mlp.backward(trainy[b:b+batch_size])
            mlp.step()

        for b in range(0, len(valx), batch_size):

            # pass  # Remove this line when you start implementing this
            # Val ...
            mlp.eval()
            mlp.zero_grads()
            mlp(valx[b:b+batch_size])
            cur_validation_errors.append(mlp.error(valy[b:b+batch_size])/batch_size)
            cur_validation_loss.append(mlp.total_loss(valy[b:b+batch_size])/batch_size)
        
        # Accumulate data...
        training_losses[e] = sum(cur_training_loss)/len(cur_training_loss)
        training_errors[e] = sum(cur_training_error)/len(cur_training_error)
        validation_losses[e] = sum(cur_validation_loss)/len(cur_validation_loss)
        validation_errors[e] = sum(cur_validation_errors)/len(cur_validation_errors)

        # Cleanup ...
        initial_indices = np.arange(len(trainx))
        np.random.shuffle(initial_indices)
        trainx = trainx[initial_indices]
        trainy = trainy[initial_indices]

        print(training_losses[e], training_errors[e], validation_losses[e], validation_errors[e])

    # Return results ...

    return (training_losses, training_errors, validation_losses, validation_errors)
    # raise NotImplemented
