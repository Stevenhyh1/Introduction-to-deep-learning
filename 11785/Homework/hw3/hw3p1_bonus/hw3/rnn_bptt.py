import numpy as np
import sys

sys.path.append('mytorch')
from rnn_cell import *
from linear import *

# RNN Phoneme Seq-to-Seq
class RNN_Phoneme_BPTT(object):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        #-------------------------------------------->
        # Don't Need Modify
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.rnn = [RNN_Cell(input_size, hidden_size) if i == 0 else
                    RNN_Cell(hidden_size, hidden_size) for i in range(num_layers)]
        self.output_layer = Linear(hidden_size, output_size)

        # store hidden states at each time step, [(seq_len+1) * (num_layers, batch_size, hidden_size)]
        self.hiddens = []
        #<---------------------------------------------

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

    def forward(self, x, x_lens, h_0=None):

        """
        RNN forward, multiple layers, multiple time steps

        Parameters
        ----------
        x : (batch_size, seq_len, input_size)
            Input with padded form
        x_lens: (batch_size, )
            Input length
        h_0 : (num_layers, batch_size, hidden_size)
            Initial hidden states. Defaults to zeros if not specified

        Returns
        -------
        out : (batch_size, seq_len, output_size)
            Output logits in padded form
        out_lens: (batch_size, )
            Output length
        """

        #-------------------------------------------->
        # Get the batch size and sequence length, and initialize the hidden
        # vectors given the paramters.
        batch_size, seq_len = x.shape[0], x.shape[1]
        output_size = self.output_size
        if h_0 is None:
            hidden = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=float)
        else:
            hidden = h_0

        # Save x and append the hidden vector to the hiddens list
        # store outputs at each time step, [batch_size * (seq_len, output_size)]
        out = np.zeros((batch_size, seq_len, output_size))
        self.x = x
        self.hiddens.append(hidden.copy())
        #<---------------------------------------------

        ### Add your code here --->
        # (More specific pseudocode may exist in lecture slides)
        # Iterate through the sequence
            # Iterate over the length of self.rnn (through the layers)
                # Run the rnn cell with the correct parameters and update
                    # the parameters as needed. Update hidden.
            # Similar to above, append a copy of the current hidden array to the hiddens list

            # Get the output of last hidden layer and feed it into output layer
            # Save current step output

        # Return output and output length
        # import pdb; pdb.set_trace()
        for t in range(seq_len):
            new_hidden = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=float)
            for b in range(batch_size):
                cur_x = x[b, t, :]
                for i in range(self.num_layers):
                    rnn_layer = self.rnn[i]
                    cur_x = rnn_layer(cur_x, self.hiddens[-1][i, b, :])
                    new_hidden[i, b] = cur_x
                cur_output = self.output_layer(cur_x)
                out[b, t, :] = cur_output
            self.hiddens.append(new_hidden)
        out_lens = x_lens
        # raise NotImplementedError
        # <--------------------------

        return out, out_lens

    def backward(self, delta, delta_lens):

        """
        RNN Back Propagation Through Time (BPTT) after CTC Loss

        Parameters
        ----------
        delta : (batch_size, seq_lens, output_size)
        gradient w.r.t. each time step output dY(i), i = 0, ...., seq_len - 1

        delta_lens : (batch_size)
        sequence length for each sample

        Returns
        -------
        dh_0 : (num_layers, batch_size, hidden_size)
        gradient w.r.t. the initial hidden states
        """

        #-------------------------------------------->
        # Initilizations
        batch_size, seq_len = self.x.shape[0], self.x.shape[1]
        dh_0 = np.zeros((self.num_layers, batch_size, self.hidden_size))
        #<---------------------------------------------


        ### Add your code here --->

        '''
        Pseudocode:
        * Get delta mask from delta_lens with 0 and 1
            delta_mask * delta sets gradient at padding time steps to 0
        * Iterate from 0 to batch_size - 1
            * Iterate in reverse order time (from b_th seq_len - 1 to 0)
                * Get dh[-1] from backward from output layer
                * Iterate in reverse order of layers (from num_layer - 1 to 0)
                    * Get h_prev_l either from hiddens or x depending on the layer
                        (Recall that hiddens has an extra initial hidden state)
                    * Use dh and hiddens to get the other parameters for the backward method
                        (Recall that hiddens has an extra initial hidden state)
                    * Update dh with the new dh from the backward pass of the rnn cell
                    * If you aren't at the first layer, you will want to add
                        dx to the gradient from l-1th layer.
            * Save dh_0 at current b_th sample
        '''

        # Attention: For Linear output layer backward, "derivative" function is added 
        #            to compute with given delta and input x 
        #            (same thing as Tanh.derivative(state = None))

        batch_size, seq_lens, output_size = delta.shape
        mask = np.arange(seq_lens).reshape(1, -1) >= delta_lens.reshape(-1, 1)
        mask = np.repeat(mask.reshape(batch_size, seq_lens, 1), output_size, axis=2)
        delta[np.where(mask == True)] = 0

        # for b in range(batch_size):
        dx = 0
        for t in range(seq_lens-1, -1, -1):
            # import pdb; pdb.set_trace()
            cur_dh = delta[:, t, :]
            cur_x = self.hiddens[t+1][-1, :, :]
            linear_dh = self.output_layer.derivative(cur_dh, cur_x)
            dh_0[-1] += linear_dh

            hidden_cur = self.hiddens[t+1]
            hidden_prev = self.hiddens[t]
            for i in range(self.num_layers-1, -1, -1):

                if i > 0:
                    h_prev_l = hidden_cur[i-1, :]
                else:
                    h_prev_l = self.x[:, t, :]

                h_prev_t = hidden_prev[i, :]

                h = hidden_cur[i, :]

                if i == 0:
                    cur_delta = dx + dh_0[i, :]
                else:
                    cur_delta = dh_0[i, :]
                dx, dh_0[i] = self.rnn[i].backward(cur_delta, h, h_prev_l, h_prev_t)

        # import pdb; pdb.set_trace()
        #<---------------------------------------------

        return dh_0



