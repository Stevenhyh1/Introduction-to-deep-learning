import json
import sys
import traceback
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

from test import *

sys.path.append('mytorch')
from conv import *
from pool import *


############################################################################################
###############################   Section 2 - Conv2D  ######################################
############################################################################################

def conv2d_forward_correctness():
    '''
    lecture 9: pg 102
    lectur 10: pg 82
    CNN: scanning with a MLP with stride
    '''
    scores_dict = [0]
    
    ############################################################################################
    #############################   Initialize parameters    ###################################
    ############################################################################################
    in_c = np.random.randint(5,15)
    out_c = np.random.randint(5,15)
    kernel = np.random.randint(3,7)
    stride = np.random.randint(3,5)
    width = np.random.randint(60,80)
    batch = np.random.randint(1,4)

    x = np.random.randn(batch, in_c, width, width)

    #############################################################################################
    #################################    Create Models   ########################################
    #############################################################################################
    def random_normal_weight_init_fn(out_channel, in_channel, kernel_width, kernel_height):
        return np.random.normal(0, 1.0, (out_channel, in_channel, kernel_width, kernel_height))
    
    test_model = Conv2D(in_c, out_c, kernel, stride, random_normal_weight_init_fn, np.zeros)
    
    torch_model = nn.Conv2d(in_c, out_c, kernel, stride=stride)
    torch_model.weight = nn.Parameter(torch.tensor(test_model.W))
    torch_model.bias = nn.Parameter(torch.tensor(test_model.b))

    #############################################################################################
    #########################    Get the correct results from PyTorch   #########################
    #############################################################################################
    x1 = Variable(torch.tensor(x), requires_grad=True)
    y1 = torch_model(x1)
    torch_y = y1.detach().numpy()
    
    #############################################################################################
    ###################    Get fwd results from TestModel and compare  ##########################
    #############################################################################################
    try:
        y2 = test_model(x)
    except NotImplementedError:
        print("Not Implemented...")
        return scores_dict
    test_y = y2

    if not assertions(test_y, torch_y, 'type', 'y'): return scores_dict
    if not assertions(test_y, torch_y, 'shape', 'y'): return scores_dict
    if not assertions(test_y, torch_y, 'closeness', 'y'): return scores_dict
    scores_dict[0] = 1
    
    return scores_dict

def test_conv2d_forward():
    np.random.seed(11785)
    n = 2
    for i in range(n):
        a = conv2d_forward_correctness()[0]
        if a != 1:
            if __name__ == '__main__':
                print('Failed Conv2D Forward Test: %d / %d' % (i + 1, n))
            return False
        else:
            if __name__ == '__main__':
                print('Passed Conv2D Forward Test: %d / %d' % (i + 1, n))
    return True

def conv2d_backward_correctness():
    '''
    lecture 9: pg 102
    lectur 10: pg 82
    CNN: scanning with a MLP with stride
    '''
    scores_dict = [0, 0, 0, 0]
    
    ############################################################################################
    #############################   Initialize parameters    ###################################
    ############################################################################################
    in_c = np.random.randint(5,15)
    out_c = np.random.randint(5,15)
    kernel = np.random.randint(3,7)
    stride = np.random.randint(3,5)
    width = np.random.randint(60,80)
    batch = np.random.randint(1,4)

    x = np.random.randn(batch, in_c, width, width)

    #############################################################################################
    #################################    Create Models   ########################################
    #############################################################################################
    def random_normal_weight_init_fn(out_channel, in_channel, kernel_width, kernel_height):
        return np.random.normal(0, 1.0, (out_channel, in_channel, kernel_width, kernel_height))
    
    test_model = Conv2D(in_c, out_c, kernel, stride, random_normal_weight_init_fn, np.zeros)
    
    torch_model = nn.Conv2d(in_c, out_c, kernel, stride=stride)
    torch_model.weight = nn.Parameter(torch.tensor(test_model.W))
    torch_model.bias = nn.Parameter(torch.tensor(test_model.b))

    #############################################################################################
    #########################    Get the correct results from PyTorch   #########################
    #############################################################################################
    x1 = Variable(torch.tensor(x), requires_grad=True)
    y1 = torch_model(x1)
    torch_y = y1.detach().numpy()
    
    b, c, w, h = torch_y.shape
    delta = np.random.randn(b, c, w, h)
    y1.backward(torch.tensor(delta))
    dy1 = x1.grad
    torch_dx = dy1.detach().numpy()
    torch_dW = torch_model.weight.grad.detach().numpy()
    torch_db = torch_model.bias.grad.detach().numpy()

    #############################################################################################
    ###################    Get fwd results from TestModel and compare  ##########################
    #############################################################################################
    try:
        y2 = test_model(x)
    except NotImplementedError:
        print("Not Implemented...")
        return scores_dict
    test_y = y2

    if not assertions(test_y, torch_y, 'type', 'y'): return scores_dict
    if not assertions(test_y, torch_y, 'shape', 'y'): return scores_dict
    if not assertions(test_y, torch_y, 'closeness', 'y'): return scores_dict
    scores_dict[0] = 1
    
    #############################################################################################
    ###################    Get bwd results from TestModel and compare  ##########################
    #############################################################################################
    try:
        dy2 = test_model.backward(delta)
    except NotImplementedError:
        print("Not Implemented...")
        return scores_dict
    test_dx = dy2
    test_dW = test_model.dW
    test_db = test_model.db
    
    if not assertions(test_dx, torch_dx, 'type', 'dx'): return scores_dict
    if not assertions(test_dx, torch_dx, 'shape', 'dx'): return scores_dict
    if not assertions(test_dx, torch_dx, 'closeness', 'dx'): return scores_dict
    scores_dict[1] = 1
    
    if not assertions(test_dW, torch_dW, 'type', 'dW'): return scores_dict
    if not assertions(test_dW, torch_dW, 'shape', 'dW'): return scores_dict
    if not assertions(test_dW, torch_dW, 'closeness', 'dW'): return scores_dict
    scores_dict[2] = 1
    
    if not assertions(test_db, torch_db, 'type', 'db'): return scores_dict
    if not assertions(test_db, torch_db, 'shape', 'db'): return scores_dict
    if not assertions(test_db, torch_db, 'closeness', 'db'): return scores_dict
    scores_dict[3] = 1
    
    #############################################################################################
    ##############################    Compare Results   #########################################
    #############################################################################################
    
    return scores_dict

def test_conv2d_backward():
    np.random.seed(11785)
    n = 2
    for i in range(n):
        a, b, c, d = conv2d_backward_correctness()
        if a != 1:
            if __name__ == '__main__':
                print('Failed Conv2D Forward Test: %d / %d' % (i + 1, n))
            return False
        elif b != 1 or c != 1 or d != 1:
            if __name__ == '__main__':
                print('Failed Conv2D Backward Test: %d / %d' % (i + 1, n))
            return False
        else:
            if __name__ == '__main__':
                print('Passed Conv2D Backward Test: %d / %d' % (i + 1, n))
    return True

############################################################################################
###############################   Section 3 - MaxPool  #####################################
############################################################################################

def max_pool_correctness():
    '''
    lecture 10: pg 42, pg 164, pg 165
    Max Pooling Layer
    '''
    scores_dict = [0, 0]
    
    ############################################################################################
    #############################   Initialize parameters    ###################################
    ############################################################################################
    kernel = np.random.randint(3,7)
    stride = np.random.randint(3,5)
    width = np.random.randint(50,100)
    in_c = np.random.randint(5,15)
    batch = np.random.randint(1,4)

    x = np.random.randn(batch, in_c, width, width)

    #############################################################################################
    #################################    Create Models   ########################################
    #############################################################################################
    torch_max_pool = nn.MaxPool2d(kernel, stride, return_indices=True)
    torch_max_unpool = nn.MaxUnpool2d(kernel, stride)

    test_model = MaxPoolLayer(kernel, stride)

    #############################################################################################
    #########################    Get the correct results from PyTorch   #########################
    #############################################################################################
    x1 = torch.tensor(x)
    y1, indices = torch_max_pool(x1)
    torch_y = y1.detach().numpy()
    x1p = torch_max_unpool(y1, indices, output_size=x1.shape)
    torch_xp = x1p.detach().numpy()

    #############################################################################################
    ###################    Get fwd results from TestModel and compare  ##########################
    #############################################################################################
    try:
        y2 = test_model(x)
    except NotImplementedError:
        print("Not Implemented...")
        return scores_dict
    test_y = y2
    
    if not assertions(test_y, torch_y, 'type', 'y'): return scores_dict
    if not assertions(test_y, torch_y, 'shape', 'y'): return scores_dict
    if not assertions(test_y, torch_y, 'closeness', 'y'): return scores_dict
    scores_dict[0] = 1
    
    #############################################################################################
    ###################    Get bwd results from TestModel and compare  ##########################
    #############################################################################################
    try:
        x2p = test_model.backward(y2)
    except NotImplementedError:
        print("Not Implemented...")
        return scores_dict
    test_xp = x2p

    if not assertions(test_xp, torch_xp, 'type', 'dx'): return scores_dict
    if not assertions(test_xp, torch_xp, 'shape', 'dx'): return scores_dict
    if not assertions(test_xp, torch_xp, 'closeness', 'dx'): return scores_dict
    # import pdb; pdb.set_trace()
    scores_dict[1] = 1

    return scores_dict

def test_max_pool():
    np.random.seed(11785)
    n = 3
    for i in range(n):
        a, b = max_pool_correctness()
        if a != 1:
            if __name__ == '__main__':
                print('Failed MaxPool Forward Test: %d / %d' % (i + 1, n))
            return False
        elif b != 1:
            if __name__ == '__main__':
                print('Failed MaxPool Backward Test: %d / %d' % (i + 1, n))
            return False
        else:
            if __name__ == '__main__':
                print('Passed MaxPool Test: %d / %d' % (i + 1, n))
    return True


############################################################################################
###############################   Section 4 - MeanPool  ####################################
############################################################################################

def mean_pool_correctness():
    '''
    lecture 10: pg 44, pg 168, pg 169
    Mean Pooling Layer
    '''
    scores_dict = [0, 0]
    
    ############################################################################################
    #############################   Initialize parameters    ###################################
    ############################################################################################
    kernel = np.random.randint(3,7)
    stride = np.random.randint(3,5)
    width = np.random.randint(50,100)
    in_c = np.random.randint(5,15)
    batch = np.random.randint(1,4)

    x = np.random.randn(batch, in_c, width, width)

    #############################################################################################
    #################################    Create Models   ########################################
    #############################################################################################
    torch_model = nn.functional.avg_pool2d
    
    test_model = MeanPoolLayer(kernel, stride)

    #############################################################################################
    #########################    Get the correct results from PyTorch   #########################
    #############################################################################################
    x1 = Variable(torch.tensor(x), requires_grad=True)
    y1 = torch_model(x1, kernel, stride)
    torch_y = y1.detach().numpy()
    y1.backward(y1)
    x1p = x1.grad
    torch_xp = x1p.detach().numpy()

    #############################################################################################
    ###################    Get fwd results from TestModel and compare  ##########################
    #############################################################################################
    try:
        y2 = test_model(x)
    except NotImplementedError:
        print("Not Implemented...")
        return scores_dict
    test_y = y2
    
    if not assertions(test_y, torch_y, 'type', 'y'): return scores_dict
    if not assertions(test_y, torch_y, 'shape', 'y'): return scores_dict
    if not assertions(test_y, torch_y, 'closeness', 'y'): return scores_dict
    scores_dict[0] = 1
    
    #############################################################################################
    ###################    Get bwd results from TestModel and compare  ##########################
    #############################################################################################
    try:
        x2p = test_model.backward(y2)
    except NotImplementedError:
        print("Not Implemented...")
        return scores_dict
    test_xp = x2p

    if not assertions(test_xp, torch_xp, 'type', 'dx'): return scores_dict
    if not assertions(test_xp, torch_xp, 'shape', 'dx'): return scores_dict
    if not assertions(test_xp, torch_xp, 'closeness', 'dx'): return scores_dict
    scores_dict[1] = 1
    
    return scores_dict

def test_mean_pool():
    n = 3
    np.random.seed(11785)
    for i in range(n):
        a, b = mean_pool_correctness()
        if a != 1:
            if __name__ == '__main__':
                print('Failed MeanPool Forward Test: %d / %d' % (i + 1, n))
            return False
        elif b != 1:
            if __name__ == '__main__':
                print('Failed MeanPool Backward Test: %d / %d' % (i + 1, n))
            return False
        else:
            if __name__ == '__main__':
                print('Passed MeanPool Test: %d / %d' % (i + 1, n))
    return True


############################################################################################
#################################### DO NOT EDIT ###########################################
############################################################################################

if __name__ == '__main__':
    
    tests = [
        {
            'name': 'Section 2 - Conv2D Forward',
            'autolab': 'Conv2D Forward',
            'handler': test_conv2d_forward,
            'value': 2,
        },
        {
            'name': 'Section 2 - Conv2D Backward',
            'autolab': 'Conv2D Backward',
            'handler': test_conv2d_backward,
            'value': 3,
        },
        {
            'name': 'Section 3 - MaxPool',
            'autolab': 'MaxPool',
            'handler': test_max_pool,
            'value': 2.5,
        },
        {
            'name': 'Section 4 - MeanPool',
            'autolab': 'MeanPool',
            'handler': test_mean_pool,
            'value': 2.5,
        },
    ]

    scores = {}
    for t in tests:
        print_name(t['name'])
        res = t['handler']()
        print_outcome(t['autolab'], res)
        scores[t['autolab']] = t['value'] if res else 0

    print(json.dumps({'scores': scores}))
