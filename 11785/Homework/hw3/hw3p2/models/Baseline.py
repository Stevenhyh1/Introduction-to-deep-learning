import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiLSTM(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layer, dropout, bidirectional=True):
        super(BiLSTM,self).__init__()

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layer,bias=True, dropout=dropout, bidirectional=bidirectional)
        if bidirectional:
            self.linear = nn.Linear(hidden_dim*2, output_dim)
        else:
            self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, lengths):
        packed_x = pack_padded_sequence(x, lengths, enforce_sorted=False)
        packed_out, (_, _) = self.lstm(packed_x)
        out, out_lens = pad_packed_sequence(packed_out)
        out = self.linear(out)
        out = F.log_softmax(out, dim=2)

        return out, out_lens

if __name__ == "__main__":
    a = torch.randn((158, 8, 40))
    model = BiLSTM(40, 256, 47)
    output = model(a)
    print(output.shape)
        
        
