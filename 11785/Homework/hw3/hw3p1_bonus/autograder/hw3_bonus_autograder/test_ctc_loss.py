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

sys.path.append('mytorch')
from ctcloss import *

# DO NOT CHANGE -->
SEED = 8888
np.random.seed(SEED)

data_path = os.path.join('autograder', 'hw3_bonus_autograder', 'data')
ref_data_path = os.path.join('autograder', 'hw3_bonus_autograder', 'data', 'ref_data')


#################################################################################################
################################   Section 1 - CTC Loss    ######################################
#################################################################################################

def test_ctc_extend_seq():
    # Get curr data
    probs = np.load(os.path.join(data_path, 'X.npy'))
    targets = np.load(os.path.join(data_path, 'Y.npy'))
    input_lens = np.load(os.path.join(data_path, 'X_lens.npy'))
    out_lens = np.load(os.path.join(data_path, 'Y_lens.npy'))

    CTC_user = CTCLoss(blank = 0)

    f_ref_S_ext = open(os.path.join(ref_data_path, 'ref_S_ext.pkl'), 'rb')
    f_ref_Skip_Connect = open(os.path.join(ref_data_path, 'ref_Skip_Connect.pkl'), 'rb')

    ref_S_ext_ls = pickle.load(f_ref_S_ext)
    ref_Skip_Connect_ls = pickle.load(f_ref_Skip_Connect)

    _, B, _ = probs.shape
    for b in range(B):
        target = targets[b, :out_lens[b]]

        user_S_ext, user_Skip_Connect = CTC_user._ext_seq_blank(target)
        user_S_ext, user_Skip_Connect = np.array(user_S_ext), np.array(user_Skip_Connect)

        ref_S_ext = ref_S_ext_ls[b]
        ref_Skip_Connect = ref_Skip_Connect_ls[b]

        if not assertions(user_S_ext, ref_S_ext, 'type', 'S_ext'): return False
        if not assertions(user_S_ext, ref_S_ext, 'shape', 'S_ext'): return False
        if not assertions(user_S_ext, ref_S_ext, 'closeness', 'S_ext'): return False
        
        if not assertions(user_Skip_Connect, ref_Skip_Connect, 'type', 'Skip_Connect'): return False
        if not assertions(user_Skip_Connect, ref_Skip_Connect, 'shape', 'Skip_Connect'): return False
        if not assertions(user_Skip_Connect, ref_Skip_Connect, 'closeness', 'Skip_Connect'): return False
    
    f_ref_S_ext.close()
    f_ref_Skip_Connect.close()
    
    return True

def ctc_posterior_prob_correctness():
    # Get curr data
    probs = np.load(os.path.join(data_path, 'X.npy'))
    targets = np.load(os.path.join(data_path, 'Y.npy'))
    input_lens = np.load(os.path.join(data_path, 'X_lens.npy'))
    out_lens = np.load(os.path.join(data_path, 'Y_lens.npy'))

    CTC_user = CTCLoss(blank = 0)

    f_ref_alpha = open(os.path.join(ref_data_path, 'ref_alpha.pkl'), 'rb')
    f_ref_beta = open(os.path.join(ref_data_path, 'ref_beta.pkl'), 'rb')
    f_ref_gamma = open(os.path.join(ref_data_path, 'ref_gamma.pkl'), 'rb')

    ref_alpha_ls = pickle.load(f_ref_alpha)
    ref_beta_ls = pickle.load(f_ref_beta)
    ref_gamma_ls = pickle.load(f_ref_gamma)

    _, B, _ = probs.shape
    for b in range(B):
        logit = probs[:input_lens[b], b]
        target = targets[b, :out_lens[b]]
        
        user_S_ext, user_Skip_Connect = CTC_user._ext_seq_blank(target)
        user_alpha = CTC_user._forward_prob(logit, user_S_ext, user_Skip_Connect)
        user_beta = CTC_user._backward_prob(logit, user_S_ext, user_Skip_Connect)
        user_gamma = CTC_user._post_prob(user_alpha, user_beta)

        ref_alpha = ref_alpha_ls[b]
        ref_beta = ref_beta_ls[b]
        ref_gamma = ref_gamma_ls[b]

        if not assertions(user_alpha, ref_alpha, 'type', 'alpha'): return 1
        if not assertions(user_alpha, ref_alpha, 'shape', 'alpha'): return 1
        if not assertions(user_alpha, ref_alpha, 'closeness', 'alpha'): return 1

        if not assertions(user_beta, ref_beta, 'type', 'beta'): return 2
        if not assertions(user_beta, ref_beta, 'shape', 'beta'): return 2
        if not assertions(user_beta, ref_beta, 'closeness', 'beta'): return 2

        if not assertions(user_gamma, ref_gamma, 'type', 'gamma'): return 3
        if not assertions(user_gamma, ref_gamma, 'shape', 'gamma'): return 3
        if not assertions(user_gamma, ref_gamma, 'closeness', 'gamma'): return 3

    f_ref_alpha.close()
    f_ref_beta.close()
    f_ref_gamma.close()
    
    return 0

def test_ctc_posterior_prob():
    b = ctc_posterior_prob_correctness()
    if b == 1:
        raise ValueError('Forward Alpha: ', 'FAIL')
    elif b == 2:
        raise ValueError('Backward Beta: ', "FAIL")
    elif b == 3:
        raise ValueError('Gamma: ', "FAIL")
    return True

def test_ctc_forward():
    # Get curr data
    probs = np.load(os.path.join(data_path, 'X.npy'))
    targets = np.load(os.path.join(data_path, 'Y.npy'))
    input_lens = np.load(os.path.join(data_path, 'X_lens.npy'))
    out_lens = np.load(os.path.join(data_path, 'Y_lens.npy'))

    CTC_user = CTCLoss(blank = 0)
    user_loss = CTC_user(probs, targets, input_lens, out_lens)

    ref_loss = np.load(os.path.join(ref_data_path, 'ref_loss.npy'))

    if not assertions(user_loss, ref_loss, 'closeness', 'forward'): return False
    
    return True

def test_ctc_backward():
    # Get curr data
    probs = np.load(os.path.join(data_path, 'X.npy'))
    targets = np.load(os.path.join(data_path, 'Y.npy'))
    input_lens = np.load(os.path.join(data_path, 'X_lens.npy'))
    out_lens = np.load(os.path.join(data_path, 'Y_lens.npy'))

    CTC_user = CTCLoss(blank = 0)
    user_loss = CTC_user(probs, targets, input_lens, out_lens)
    user_dy = CTC_user.derivative()

    ref_dy = np.load(os.path.join(ref_data_path, 'ref_dy.npy'))

    if not assertions(user_dy, ref_dy, 'type', 'backward'): return False
    if not assertions(user_dy, ref_dy, 'closeness', 'backward'): return False
    
    return True
