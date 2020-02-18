import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self,input_dimension,output_dimension):
        super(CNN, self).__init__()

    def forward(self,x):
        return x