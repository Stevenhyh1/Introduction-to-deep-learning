import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self,input_dimension,output_dimension):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dimension, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.dp1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(4096,2048)
        self.bn2 = nn.BatchNorm1d(2048)
        self.dp2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(2048,1024)
        self.fc4 = nn.Linear(1024,1024)
        self.fc5 = nn.Linear(1024,output_dimension)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dp1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dp2(x)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


