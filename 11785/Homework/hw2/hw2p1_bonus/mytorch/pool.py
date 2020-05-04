import numpy as np

class MaxPoolLayer():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride
        self.x = None
        self.pidx = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
            out (np.array): (batch_size, out_channel, output_width, output_height)
        """
        self.x = x
        batch_size, in_channel, input_width, input_height = x.shape
        output_width = int(np.floor((input_width - self.kernel)/self.stride) + 1)
        output_height = int(np.floor((input_height - self.kernel)/self.stride) + 1)
        out_channel = in_channel
        output = np.zeros((batch_size, out_channel, output_width, output_height))
        self.pidx = np.zeros((batch_size, in_channel, output_width, output_height), dtype=np.int64)
        for b in range(batch_size):
            for i in range(output_height):
                for j in range(output_width):
                    #  segment: (in_channel, kernel, kernel)
                    h_start = i*self.stride
                    h_end = i*self.stride+self.kernel
                    w_start = j*self.stride
                    w_end = j*self.stride+self.kernel
                    segment = x[b, :, h_start:h_end, w_start:w_end]
                    maxele = np.amax(segment, axis=(1, 2))  # (out_channel, )
                    output[b, :, i, j] = maxele

                    flatten_seg = segment.reshape(in_channel, -1)  # (in_channel, kernel*kernel)
                    max_idx = flatten_seg.argmax(1)
                    self.pidx[b, :, i, j] = max_idx
        return output

        # raise NotImplementedError

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_width, output_height)
        Return:
            dx (np.array): (batch_size, in_channel, input_width, input_height)
        """
        # import pdb; pdb.set_trace()
        dx = np.zeros_like(self.x)
        batch_size, in_channel, output_width, output_height = delta.shape
        for b in range(batch_size):
            for i in range(output_height):
                for j in range(output_width):

                    h_start = i*self.stride
                    h_end = i*self.stride+self.kernel
                    w_start = j*self.stride
                    w_end = j*self.stride+self.kernel

                    cur_dz = np.tile(delta[b, :, i, j], (self.kernel*self.kernel, 1)).T
                    max_idx = self.pidx[b, :, i, j]
                    mask = np.arange(self.kernel * self.kernel).reshape(1, -1) == max_idx.reshape(-1, 1)
                    cur_pidx = np.zeros((in_channel, self.kernel*self.kernel), dtype=np.int64)
                    cur_pidx[np.where(mask == True)] = 1
                    cur_dx = (cur_pidx*cur_dz).reshape(in_channel, self.kernel, self.kernel)
                    cur_dx[np.where(dx[b, :, h_start:h_end, w_start:w_end] != 0)] = 0
                    dx[b, :, h_start:h_end, w_start:w_end] += cur_dx
        return dx

class MeanPoolLayer():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride
        self.x = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_width, input_height)
        Return:
            out (np.array): (batch_size, out_channel, output_width, output_height)
        """
        self.x = x
        batch_size, in_channel, input_width, input_height = x.shape
        output_width = int(np.floor((input_width - self.kernel)/self.stride) + 1)
        output_height = int(np.floor((input_height - self.kernel)/self.stride) + 1)
        out_channel = in_channel
        output = np.zeros((batch_size, out_channel, output_width, output_height))

        for b in range(batch_size):
            for i in range(output_height):
                for j in range(output_width):
                    #  segment: (in_channel, kernel, kernel)
                    h_start = i*self.stride
                    h_end = i*self.stride+self.kernel
                    w_start = j*self.stride
                    w_end = j*self.stride+self.kernel
                    segment = x[b, :, h_start:h_end, w_start:w_end]
                    mean = np.mean(segment, axis=(1, 2))  # (out_channel, )
                    output[b, :, i, j] = mean

        return output

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_width, output_height)
        Return:
            dx (np.array): (batch_size, in_channel, input_width, input_height)
        """
        dx = np.zeros_like(self.x)
        batch_size, in_channel, output_width, output_height = delta.shape
        for b in range(batch_size):
            for i in range(output_height):
                for j in range(output_width):

                    h_start = i*self.stride
                    h_end = i*self.stride+self.kernel
                    w_start = j*self.stride
                    w_end = j*self.stride+self.kernel

                    cur_dz = np.tile(delta[b, :, i, j], (self.kernel*self.kernel, 1)).T
                    cur_pidx = np.ones((in_channel, self.kernel*self.kernel))/(self.kernel*self.kernel)
                    cur_dx = (cur_pidx*cur_dz).reshape(in_channel, self.kernel, self.kernel)
                    dx[b, :, h_start:h_end, w_start:w_end] += cur_dx
        return dx
