import random
import numpy as np
import os
import sys
import time
import pickle
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

sys.path.append('autograder')
from helpers import *

sys.path.append('hw3')
from rnn_bptt import RNN_Phoneme_BPTT

ref_data_path = os.path.join('autograder', 'hw3_bonus_autograder', 'data', 'ref_data')


#################################################################################################
################################   Section 2 - RNN BPTT    ######################################
#################################################################################################

def test_rnn_bptt_fwd():
    rnn_layers = 2
    batch_size = 5
    seq_len = 30
    input_size = 40
    hidden_size = 32 # hidden_size > 100 will cause precision error
    output_size = 47

    np.random.seed(11785)

    data_x = np.random.randn(batch_size, seq_len, input_size)
    data_x_lens = np.random.randint(25, high = 29, size = batch_size, dtype = 'int')
    for b in range(batch_size):
        data_x[b, data_x_lens[b]:] = 0
    data_y = np.random.randint(0, output_size, size = (batch_size, 5))
    data_y_lens = np.random.randint(2, 4, size = batch_size)
    for b in range(batch_size):
        data_y[b, data_y_lens[b]:] = 0

    f_fc_Ws = open(os.path.join(ref_data_path, 'fc_Ws.pkl'), 'rb')
    f_rnn_Ws = open(os.path.join(ref_data_path, 'rnn_Ws.pkl'), 'rb')

    # Initialize
    # Reference model
    # ref_rnn_model = RNN_Phoneme_BPTT_Ref(input_size, hidden_size, output_size, num_layers = rnn_layers)
    
    # My model
    my_rnn_model = RNN_Phoneme_BPTT(input_size, hidden_size, output_size, num_layers = rnn_layers)

    # rnn_weights = [[ref_rnn_model.rnn[l].W_ih,
    #             ref_rnn_model.rnn[l].W_hh,
    #             ref_rnn_model.rnn[l].b_ih,
    #             ref_rnn_model.rnn[l].b_hh,] for l in range(rnn_layers)]
    
    # fc_weights = [ref_rnn_model.output_layer.W, ref_rnn_model.output_layer.b]

    rnn_weights = pickle.load(f_rnn_Ws)
    fc_weights = pickle.load(f_fc_Ws)
    f_rnn_Ws.close()
    f_fc_Ws.close()

    my_rnn_model.init_weights(rnn_weights, fc_weights)

    # Test forward pass
    # Reference model
    # ref_out, ref_out_lens = ref_rnn_model(data_x, data_x_lens)
    
    # My model
    my_out, my_out_lens = my_rnn_model(data_x, data_x_lens)

    # np.save(os.path.join(ref_data_path,'ref_out.npy'), ref_out)
    # np.save(os.path.join(ref_data_path, 'ref_out_lens.npy'), ref_out_lens)

    ref_out = np.load(os.path.join(ref_data_path,'ref_out.npy'))
    ref_out_lens = np.load(os.path.join(ref_data_path, 'ref_out_lens.npy'))

    # Verify forward outputs
    # print('Testing RNN Seq-to-Seq Forward...')
    try:
        assert(np.allclose(my_out, ref_out, rtol=1e-03))
        assert(np.allclose(my_out_lens, ref_out_lens, rtol = 1e-03))
    except:
        raise ValueError('RNN Forward: FAIL')
    # if not self.assertions(my_out, ref_out, 'closeness', 'RNN Classifier Forwrd'): #rtol=1e-03)
        # return 'RNN Forward'
    # print('RNN Forward: PASS' )

    return True

def test_rnn_bptt_bwd():
    rnn_layers = 2
    batch_size = 5
    seq_len = 30
    input_size = 40
    hidden_size = 32 # hidden_size > 100 will cause precision error
    output_size = 47

    np.random.seed(11785)

    data_x = np.random.randn(batch_size, seq_len, input_size)
    data_x_lens = np.random.randint(25, high = 29, size = batch_size, dtype = 'int')
    for b in range(batch_size):
        data_x[b, data_x_lens[b]:] = 0
    data_y = np.random.randint(0, output_size, size = (batch_size, 5))
    data_y_lens = np.random.randint(2, 4, size = batch_size)
    for b in range(batch_size):
        data_y[b, data_y_lens[b]:] = 0

    # Initialize
    # Reference model
    ref_rnn_model = RNN_Phoneme_BPTT(input_size, hidden_size, output_size, num_layers = rnn_layers)
    
    # My model
    my_rnn_model = RNN_Phoneme_BPTT(input_size, hidden_size, output_size, num_layers = rnn_layers)

    # rnn_weights = [[ref_rnn_model.rnn[l].W_ih,
    #             ref_rnn_model.rnn[l].W_hh,
    #             ref_rnn_model.rnn[l].b_ih,
    #             ref_rnn_model.rnn[l].b_hh,] for l in range(rnn_layers)]
    
    # fc_weights = [ref_rnn_model.output_layer.W, ref_rnn_model.output_layer.b]

    f_fc_Ws = open(os.path.join(ref_data_path, 'fc_Ws.pkl'), 'rb')
    f_rnn_Ws = open(os.path.join(ref_data_path, 'rnn_Ws.pkl'), 'rb')
    rnn_weights = pickle.load(f_rnn_Ws)
    fc_weights = pickle.load(f_fc_Ws)
    f_rnn_Ws.close()
    f_fc_Ws.close()

    my_rnn_model.init_weights(rnn_weights, fc_weights)

    # Test forward pass
    # Reference model
    # ref_out, ref_out_lens = ref_rnn_model(data_x, data_x_lens)
    
    # My model
    my_out, my_out_lens = my_rnn_model(data_x, data_x_lens)


    # print('Testing RNN Seq-to-Seq Backward...')

    # Test backward pass
    # Reference model
    # ref_criterion = CTCLoss_Ref(blank = 0)


    # todo: use CTCLoss compute derivative delta
    # ref_out = np.transpose(ref_out, (1, 0, 2))
    # ref_loss = ref_criterion(ref_out, data_y, ref_out_lens, data_y_lens)
    # ref_delta = ref_criterion.derivative()
    # ref_delta = np.transpose(ref_delta, (1, 0, 2))
    np.random.seed(11785)
    B, T, H = my_out.shape

    delta = np.random.randn(B, T, H)

    # ref_dh = ref_rnn_model.backward(delta, data_y_lens)

    # My model
    # my_criterion = CTCLoss(blank = 0)
    # my_out = np.transpose(my_out, (1, 0, 2))
    # my_loss = my_criterion(my_out, data_y, my_out_lens, data_y_lens)
    # my_delta = my_criterion.derivative()
    # my_delta = np.transpose(my_delta, (1, 0, 2))
    my_dh = my_rnn_model.backward(delta, data_y_lens)

    # np.save(os.path.join(ref_data_path, 'ref_dh.npy'), ref_dh)
    # np.save(os.path.join(ref_data_path, 'ref_dW.npy'), ref_rnn_model.output_layer.dW)
    # np.save(os.path.join(ref_data_path, 'ref_db.npy'), ref_rnn_model.output_layer.db)

    # from collections import defaultdict
    # f_rnn_deriv = open(os.path.join(ref_data_path, 'ref_rnn_deriv.pkl'), 'wb')
    # rnn_deriv = defaultdict(list)

    # for l, rnn_cell in enumerate(my_rnn_model.rnn):
    #     rnn_deriv['dW_ih'].append(ref_rnn_model.rnn[l].dW_ih)
    #     rnn_deriv['dW_hh'].append(ref_rnn_model.rnn[l].dW_hh)
    #     rnn_deriv['db_ih'].append(ref_rnn_model.rnn[l].db_ih)
    #     rnn_deriv['db_hh'].append(ref_rnn_model.rnn[l].db_hh)
    
    # pickle.dump(rnn_deriv, f_rnn_deriv, -1)
    # f_rnn_deriv.close()

    f_rnn_deriv = open(os.path.join(ref_data_path, 'ref_rnn_deriv.pkl'), 'rb')
    ref_dh = np.load(os.path.join(ref_data_path, 'ref_dh.npy'))
    ref_dW = np.load(os.path.join(ref_data_path, 'ref_dW.npy'))
    ref_db = np.load(os.path.join(ref_data_path, 'ref_db.npy'))
    rnn_deriv = pickle.load(f_rnn_deriv)
    f_rnn_deriv.close()

    # Verify derivative w.r.t. each network parameters
    try:
        # import pdb; pdb.set_trace()
        assert(np.allclose(my_dh, ref_dh, rtol=1e-03))
        assert(np.allclose(my_rnn_model.output_layer.dW, ref_dW, rtol=1e-03))
        assert(np.allclose(my_rnn_model.output_layer.db, ref_db, rtol=1e-03))

        for l, rnn_cell in enumerate(my_rnn_model.rnn):

            assert(np.allclose(my_rnn_model.rnn[l].dW_ih, rnn_deriv['dW_ih'][l], rtol=1e-03))
            assert(np.allclose(my_rnn_model.rnn[l].dW_hh, rnn_deriv['dW_hh'][l], rtol=1e-03))
            assert(np.allclose(my_rnn_model.rnn[l].db_ih, rnn_deriv['db_ih'][l], rtol=1e-03))
            assert(np.allclose(my_rnn_model.rnn[l].db_hh, rnn_deriv['db_ih'][l], rtol=1e-03))


    except:
        raise ValueError('RNN Backward: FAIL')

    # print('RNN Seq-to-Seq Backward: PASS' )
    return True
