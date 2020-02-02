import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self,input_dimension,output_dimension):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dimension, 64)
        self.fc2 = nn.Linear(64,128)
        self.fc3 = nn.Linear(128,256)
        self.fc4 = nn.Linear(256,512)
        self.fc5 = nn.Linear(512,output_dimension)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


