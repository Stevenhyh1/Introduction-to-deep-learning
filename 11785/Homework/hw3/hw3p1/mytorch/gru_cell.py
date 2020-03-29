import numpy as np
from activation import *

class GRU_Cell:
    """docstring for GRU_Cell"""
    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x_t=0

        self.Wzh = np.random.randn(h,h)
        self.Wrh = np.random.randn(h,h)
        self.Wh  = np.random.randn(h,h)

        self.Wzx = np.random.randn(h,d)
        self.Wrx = np.random.randn(h,d)
        self.Wx  = np.random.randn(h,d)

        self.dWzh = np.zeros((h,h))
        self.dWrh = np.zeros((h,h))
        self.dWh  = np.zeros((h,h))

        self.dWzx = np.zeros((h,d))
        self.dWrx = np.zeros((h,d))
        self.dWx  = np.zeros((h,d))

        self.z_act = Sigmoid()
        self.r_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here


    def init_weights(self, Wzh, Wrh, Wh, Wzx, Wrx, Wx):
        self.Wzh = Wzh
        self.Wrh = Wrh
        self.Wh = Wh
        self.Wzx = Wzx
        self.Wrx = Wrx
        self.Wx  = Wx

    def __call__(self, x, h):
        return self.forward(x,h)

    def forward(self, x, h):
        # input:
        #   - x: shape(input dim),  observation at current time-step
        #   - h: shape(hidden dim), hidden-state at previous time-step
        #
        # output:
        #   - h_t: hidden state at current time-step

        self.x = x
        self.hidden = h

        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.
        assert self.x.shape == (self.d, )
        assert self.hidden.shape == (self.h, )
        # import pdb; pdb.set_trace()
        self.z = self.z_act(np.dot(self.Wzh,self.hidden)+np.dot(self.Wzx,self.x))
        self.r = self.r_act(np.dot(self.Wrh,self.hidden)+np.dot(self.Wrx,self.x))
        self.h_tilda = self.h_act(np.dot(self.Wh,self.r*self.hidden)+np.dot(self.Wx,self.x))
        h_t = (1-self.z)*self.hidden+self.z*self.h_tilda

        assert self.r.shape == (self.h, )
        assert self.z.shape == (self.h, )
        assert self.h_tilda.shape == (self.h, )
        assert h_t.shape == (self.h, )


        return h_t
        # raise NotImplementedError


    # This must calculate the gradients wrt the parameters and return the
    # derivative wrt the inputs, xt and ht, to the cell.
    def backward(self, delta):
        # input:
        #  - delta:  shape (hidden dim), summation of derivative wrt loss from next layer at
        #            the same time-step and derivative wrt loss from same layer at
        #            next time-step
        # output:
        #  - dx: Derivative of loss wrt the input x
        #  - dh: Derivative  of loss wrt the input hidden h

        # 1) Reshape everything you saved in the forward pass.
        # 2) Compute all of the derivatives
        # 3) Know that the autograders the gradients in a certain order, and the
        #    local autograder will tell you which gradient you are currently failing.

        self.x = self.x.reshape(1,self.d)
        self.hidden = self.hidden.reshape(1,self.h)

        d1 = (1-self.z) * delta
        d2 = self.hidden * delta
        d3 = self.h_tilda * delta
        d4 = -1*d2
        d5 = d3 + d4
        d6 = self.z * delta
        d7 = d5 * self.z_act.derivative()
        d8 = d6 * self.h_act.derivative()
        d9 = np.dot(d8, self.Wx)
        d10 = np.dot(d8, self.Wh)
        d11 = np.dot(d7, self.Wzx)
        d12 = np.dot(d7, self.Wzh)
        d14 = d10 * self.r
        d15 = d10 * self.hidden
        d16 = d15 * self.r_act.derivative()
        d13 = np.dot(d16, self.Wrx)
        d17 = np.dot(d16, self.Wrh)

        dx = d9 + d11 + d13
        dh = d12 + d14 + d1 + d17
        # import pdb; pdb.set_trace()
        self.dWrx += np.dot(d16.T,self.x)
        self.dWzx += np.dot(d7.T,self.x)
        self.dWx += np.dot(d8.T,self.x)
        self.dWrh += np.dot(d16.T,self.hidden)
        self.dWzh += np.dot(d7.T,self.hidden)
        self.dWh += np.dot(d8.T, (self.hidden*self.r))

        assert dx.shape == (1, self.d)
        assert dh.shape == (1, self.h)

        return dx, dh
        # raise NotImplementedError
