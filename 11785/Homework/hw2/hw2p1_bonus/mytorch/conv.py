import numpy as np


class Conv2D():
    def __init__(self, in_channel, out_channel,
                 kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size, kernel_size)

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
            x (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
            out (np.array): (batch_size, out_channel, output_width, output_height)
        """
        batch_size, in_channel, input_width, input_height = x.shape
        self.x = x
        self.input_width = input_width
        self.intput_height = input_height
        output_width = int(np.floor((input_width - self.kernel_size)/self.stride) + 1)
        output_height = int(np.floor((input_height - self.kernel_size)/self.stride) + 1)
        output = np.zeros((batch_size, self.out_channel, output_width, output_height))
        for b in range(batch_size):
            for c_out in range(self.out_channel):
                cur_weight = self.W[c_out]
                cur_bias = self.b[c_out]
                for i in range(output_height):
                    for j in range(output_width):
                        cur_input = x[b, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size]
                        cur_ouput = np.sum(cur_input * cur_weight)+cur_bias
                        output[b, c_out, i, j] = cur_ouput
        return output

        # raise NotImplementedError

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_width, output_height)
        Return:
            dx (np.array): (batch_size, in_channel, input_width, input_height)
        """

        batch_size, out_channel, output_width, output_height = delta.shape
        dx = np.zeros((batch_size, self.in_channel, self.input_width, self.intput_height))
        self.db = np.sum(delta, axis=(0, 2, 3))

        for b in range(batch_size):
            for c_out in range(out_channel):
                for i in range(output_height):
                    for j in range(output_width):
                        for c_in in range(self.in_channel):
                            cur_weight = self.W[c_out, c_in]  # (kernel, kernel)
                            w_start = i*self.stride
                            h_start = j*self.stride
                            for x in range(self.kernel_size):
                                for y in range(self.kernel_size):
                                    dx[b, c_in, w_start+x, h_start+y] += delta[b, c_out, i, j] * cur_weight[x, y]
                                    self.dW[c_out, c_in, x, y] += delta[b, c_out, i, j] * self.x[b, c_in, w_start+x, h_start+y]

        return dx
