# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np


class Conv1D():
    def __init__(self, in_channel, out_channel, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
       # Do not modify this method
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size)
        
        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_size)
        Return:
            out (np.array): (batch_size, out_channel, output_size)
        """
        # self.x = x
        # batch_size = x.shape[0]
        # in_width = x.shape[-1]
        # out_width = int(np.floor(1+(in_width-self.kernel_size)/self.stride))
        # out = np.zeros((batch_size,self.out_channel,out_width))
        # for n in range(batch_size):
        #     for k2 in range(self.out_channel):
        #         bias = self.b[k2]
        #         for t in range(out_width):
        #             affine = 0
        #             s = self.stride*t
        #             for k1 in range(self.in_channel):
        #                 affine += np.dot(self.W[k2][k1],x[n][k1][s:s+self.kernel_size])
        #             out[n][k2][t] = bias + affine
        # return out

        self.x = x
        batch_size = x.shape[0]
        self.in_width = x.shape[-1]
        self.out_width = (self.in_width-self.kernel_size)//self.stride+1
        out = np.zeros((batch_size,self.out_channel,self.out_width))
        w = self.W.reshape(self.out_channel,-1)
        
        for t in range(self.out_width):
            x_kernel = x[:, :, t*self.stride:t*self.stride+self.kernel_size]
            # import pdb; pdb.set_trace()
            # x_kernel = np.transpose(x_kernel, (1,2,0))
            # w = np.transpose(self.W, (2,1,0))
            # out[:,:,t] = np.tensordot(x, w, axes=([1,0],[0,1]))
            out[:,:,t] = np.dot(x_kernel.reshape(batch_size,-1), np.transpose(w))+self.b.reshape(1,-1)

        return out
        # raise NotImplemented



    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_size)
        Return:
            dx (np.array): (batch_size, in_channel, input_size)
        """

        batch_size = delta.shape[0]
        dx=np.zeros_like(self.x)
        self.db=np.sum(delta,axis=(0,2))
        self.dW = np.zeros_like(self.W) 
        # x: 2, 5, 151
        # W: 13, 5 ,4
        # delta: 2, 13, 22
        
        for t in range(self.out_width):
            x_kernel=self.x[:,:,t*self.stride:t*self.stride+self.kernel_size]     
            #x_kernel: 2,5,4
            # import pdb; pdb.set_trace()

            self.dW += np.dot(delta[:,:,t].T,x_kernel.reshape((batch_size,-1))).reshape((self.dW.shape))
            dx[:,:,t*self.stride:t*self.stride+self.kernel_size] += np.dot(delta[:,:,t],self.W.reshape((self.out_channel,-1))).reshape((x_kernel.shape))

        return dx
        # raise NotImplemented



class Flatten():
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, in_width)
        Return:
            out (np.array): (batch_size, in_channel * in width)
        """
        self.b, self.c, self.w = x.shape
        out = x.reshape(self.b, -1)
        return out
        # raise NotImplemented

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in channel * in width)
        Return:
            dx (np.array): (batch size, in channel, in width)
        """
        out = delta.reshape(self.b, self.c, self.w)
        return out
        # raise NotImplemented
