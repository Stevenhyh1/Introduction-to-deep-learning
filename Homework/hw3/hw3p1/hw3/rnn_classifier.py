import numpy as np
import sys

sys.path.append('mytorch')
from rnn_cell import *
from linear import *

# RNN Phoneme Classifier
class RNN_Phoneme_Classifier(object):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        ### TODO: Understand then uncomment this code :)
        self.rnn = [RNN_Cell(input_size, hidden_size) if i == 0 else
                    RNN_Cell(hidden_size, hidden_size) for i in range(num_layers)]
        self.output_layer = Linear(hidden_size, output_size)

        # store hidden states at each time step, [(seq_len+1) * (num_layers, batch_size, hidden_size)]
        self.hiddens = []

    def init_weights(self, rnn_weights, linear_weights):
        """
        Initialize weights

        Parameters
        ----------
        rnn_weights:
        [[W_ih_l0, W_hh_l0, b_ih_l0, b_hh_l0],
         [W_ih_l1, W_hh_l1, b_ih_l1, b_hh_l1], ...]

        linear_weights:
        [W, b]
        """
        for i, rnn_cell in enumerate(self.rnn):
            rnn_cell.init_weights(*rnn_weights[i])
        self.output_layer.init_weights(*linear_weights)

    def __call__(self, x, h_0=None):
        return self.forward(x, h_0)

    def forward(self, x, h_0=None):

        """
        RNN forward, multiple layers, multiple time steps

        Parameters
        ----------
        x : (batch_size, seq_len, input_size)
            Input
        h_0 : (num_layers, batch_size, hidden_size)
            Initial hidden states. Defaults to zeros if not specified

        Returns
        -------
        logits : (batch_size, output_size)
        Output logits
        """

        # Get the batch size and sequence length, and initialize the hidden
        # vectors given the paramters.
        batch_size, seq_len = x.shape[0], x.shape[1]
        if h_0 is None:
            hidden = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=float)
        else:
            hidden = h_0

        # Save x and append the hidden vector to the hiddens list
        self.x = x
        self.hiddens.append(hidden.copy())

        ### Add your code here --->
        # (More specific pseudocode may exist in lecture slides)
        # Iterate through the sequence
            # Iterate over the length of your self.rnn (through the layers)
                # Run the rnn cell with the correct parameters and update
                # the parameters as needed. Update hidden.
            # Similar to above, append a copy of the current hidden array to the hiddens list
        # Get the outputs from the last time step using the linear layer and return it
        # logits =

        for t in range(seq_len):
            hidden = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=float)
            h_cur = self.x[:,t,:]
            for l in range(self.num_layers):
                h_prev = self.hiddens[t][l,:,:]
                h_cur = self.rnn[l](h_cur, h_prev)
                hidden[l,:,:] = h_cur
            self.hiddens.append(hidden)
        logits = self.output_layer(self.hiddens[-1][-1,:,:])       
        return logits
        # <--------------------------
        # raise NotImplementedError

    def backward(self, delta):

        """
        RNN Back Propagation Through Time (BPTT)

        Parameters
        ----------
        delta : (batch_size, hidden_size)
        gradient w.r.t. the last time step output dY(seq_len-1)

        Returns
        -------
        dh_0 : (num_layers, batch_size, hidden_size)
        gradient w.r.t. the initial hidden states
        """

        # Initilizations
        batch_size, seq_len = self.x.shape[0], self.x.shape[1]
        dh = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=float)
        dh[-1] = self.output_layer.backward(delta)

        '''

        * Notes:
        * More specific pseudocode may exist in lecture slides and a visualization
          exists in the writeup.
        * WATCH out for off by 1 errors due to implementation decisions.

        Pseudocode:
        * Iterate in reverse order of time (from seq_len-1 to 0)
            * Iterate in reverse order of layers (from num_layers-1 to 0)
                * Get h_prev_l either from hiddens or x depending on the layer
                    (Recall that hiddens has an extra initial hidden state)
                * Use dh and hiddens to get the other parameters for the backward method
                    (Recall that hiddens has an extra initial hidden state)
                * Update dh with the new dh from the backward pass of the rnn cell
                * If you aren't at the first layer, you will want to add
                  dx to the gradient from l-1th layer.

        * Normalize dh by batch_size since initial hidden states are also treated
          as parameters of the network (divide by batch size)

        '''
        # import pdb; pdb.set_trace()
        dx = 0
        for t in range(seq_len):
            t = seq_len-1-t
            hidden_cur=self.hiddens[t+1]
            hidden_prev=self.hiddens[t]

            for l in range(self.num_layers):
                l = self.num_layers-1-l

                if l>0:
                    h_prev_l = hidden_cur[l-1,:,:]
                else:
                    h_prev_l = self.x[:,t,:]

                h_prev_t = hidden_prev[l,:,:]
                
                h = hidden_cur[l,:,:]

                if l == 0:
                    delta = dx + dh[l]
                else:
                    delta = dh[l]
                dx, dh[l] = self.rnn[l].backward(delta, h, h_prev_l, h_prev_t)


        # import pdb; pdb.set_trace()
        # [-2.99956228e-05,  1.03055936e-05, -4.48601531e-06,
        #   1.35824148e-05,  2.50065186e-05,  4.18231793e-05,
        #  -3.40834413e-05, -2.45170904e-05, -4.37613789e-06,
        #   2.86121722e-05,  1.28903025e-06,  5.77159426e-06,
        #  -9.05111028e-06, -5.73825764e-06, -1.48508619e-08,
        #   1.65372421e-05,  1.08223585e-05, -2.81397352e-05,
        #  -9.00760824e-06, -1.56276183e-05, -1.77527018e-05,
        #  -1.66199752e-05, -5.62665079e-08, -5.52370147e-06,
        #  -3.96908581e-06, -3.64228799e-05,  1.44748037e-05,
        #   5.60875469e-06,  3.87646469e-05, -3.42577587e-05,
        #   6.98853273e-06, -2.05346714e-05]
        dh_0 = dh.copy()/batch_size
        return dh_0

        # raise NotImplementedError
