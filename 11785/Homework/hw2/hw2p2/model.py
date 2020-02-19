import torch
import torch.nn as nn

class Classification(nn.Module):
    def __init__(self,input_dimension,output_dimension):
        super(Classification, self).__init__()

    def forward(self,x):
        return x