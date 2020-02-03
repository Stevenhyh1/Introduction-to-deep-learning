# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np

class BatchNorm(object):

    def __init__(self, in_feature, alpha=0.9):

        # You shouldn't need to edit anything in init

        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        # The following attributes will be tested
        self.var = np.ones((1, in_feature))
        self.mean = np.zeros((1, in_feature))

        self.gamma = np.ones((1, in_feature))
        self.dgamma = np.zeros((1, in_feature))

        self.beta = np.zeros((1, in_feature))
        self.dbeta = np.zeros((1, in_feature))

        # inference parameters
        self.running_mean = np.zeros((1, in_feature))
        self.running_var = np.ones((1, in_feature))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):
        """
        Argument:
            x (np.array): (batch_size, in_feature)
            eval (bool): inference status

        Return:
            out (np.array): (batch_size, in_feature)
        """

        self.x = x

        if eval:
            self.norm = (self.x - self.running_mean) / (self.running_var + self.eps)
            self.out = (self.norm * self.gamma + self.beta)
        
        else:
            self.mean = np.mean(self.x,axis=0,keepdims=True)
            self.var = np.var(self.x,axis=0,keepdims=True)
            self.norm = (self.x - self.mean)/np.sqrt(self.var + self.eps)
            self.out = self.gamma*self.norm  + self.beta

            # Update running batch statistics
            self.running_mean = self.mean
            self.running_var = self.var

        return self.out
        # raise NotImplemented


    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in feature)
        Return:
            out (np.array): (batch size, in feature)
        """
        
        dnorm = delta*self.gamma
        self.dbeta = np.sum(delta,axis=0,keepdims=True)
        self.dgamma = np.sum(delta*self.norm, axis=0, keepdims=True)
        dvar = -0.5*np.sum((self.x-self.mean)*np.power((self.var + self.eps),-3/2), axis = 0, keepdims = True)
        dmean = -np.sum(dnorm*np.power((self.var + self.eps),-0.5), axis = 0, keepdims = True) - 2*dvar*np.mean(self.x-self.mean, axis = 0, keepdims = True)
        m = len(self.x)
        dx = dnorm*np.power((self.var+self.eps),-0.5) + dvar*(2/m*(self.x-self.mean)) + dmean/m

        return dx
        # raise NotImplemented
