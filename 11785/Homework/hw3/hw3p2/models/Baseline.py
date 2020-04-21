import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiLSTM1(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layer, dropout, bidirectional=True):
        super(BiLSTM1,self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=512, kernel_size=5, stride=1, padding=6, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, stride=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, stride=1, bias=False),
            nn.BatchNorm1d(512)
        )
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_dim, num_layers=num_layer,bias=True, dropout=dropout, bidirectional=bidirectional)
        if bidirectional:
            self.linear = nn.Linear(hidden_dim*2, 512)
            self.bn = nn.BatchNorm1d(512)
            self.relu = nn.ReLU(inplace = True)
            self.linear2 = nn.Linear(512, output_dim)
            
        else:
            self.linear = nn.Linear(hidden_dim, 512)
            self.bn = nn.BatchNorm1d(512)
            self.relu = nn.ReLU(inplace = True)
            self.linear2 = nn.Linear(512, output_dim)

    def forward(self, x, lengths):
        # import pdb; pdb.set_trace()
        # x_cnn = x.permute(1,2,0)
        x_cnn = x.transpose(0,1).transpose(1,2).contiguous()
        x_cnn = self.cnn(x_cnn)
        # x_cnn = x_cnn.permute(2,0,1)
        x_cnn = x_cnn.transpose(1,2).transpose(0,1).contiguous()
        packed_x = pack_padded_sequence(x_cnn, lengths, enforce_sorted=False)
        packed_out, (_, _) = self.lstm(packed_x)
        out, out_lens = pad_packed_sequence(packed_out)
        #out shape: [T, B, D]
        
        # import pdb; pdb.set_trace()

        out = out.transpose(0,1).contiguous()
        # import pdb; pdb.set_trace()
        out = self.linear(out)
        out = out.transpose(1,2).contiguous()
        out = self.relu(self.bn(out))
        out = out.transpose(1,2).contiguous()
        out = self.linear2(out)
        out = out.transpose(0,1).contiguous()
        out = F.log_softmax(out, dim=2)

        return out, out_lens


class BiLSTM(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layer, dropout, bidirectional=True):
        super(BiLSTM,self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=512, kernel_size=5, stride=1, padding=6, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, stride=1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, stride=1, bias=False),
            nn.BatchNorm1d(512)
        )
        self.lstm = nn.LSTM(input_size=40, hidden_size=hidden_dim, num_layers=num_layer,bias=True, dropout=dropout, bidirectional=bidirectional)
        if bidirectional:
            self.linear = nn.Linear(hidden_dim*2, 512)
            self.bn = nn.BatchNorm1d(512)
            self.relu = nn.ReLU(inplace = True)
            self.linear2 = nn.Linear(512, output_dim)
            
        else:
            self.linear = nn.Linear(hidden_dim, 512)
            self.bn = nn.BatchNorm1d(512)
            self.relu = nn.ReLU(inplace = True)
            self.linear2 = nn.Linear(512, output_dim)

    def forward(self, x, lengths):
        # import pdb; pdb.set_trace()
        # x_cnn = x.permute(1,2,0)
        # x_cnn = x.transpose(0,1).transpose(1,2).contiguous()
        # x_cnn = self.cnn(x_cnn)
        # # x_cnn = x_cnn.permute(2,0,1)
        # x_cnn = x_cnn.transpose(1,2).transpose(0,1).contiguous()
        packed_x = pack_padded_sequence(x, lengths, enforce_sorted=False)
        packed_out, (_, _) = self.lstm(packed_x)
        out, out_lens = pad_packed_sequence(packed_out)
        #out shape: [T, B, D]
        
        # import pdb; pdb.set_trace()

        out = out.transpose(0,1).contiguous()
        # import pdb; pdb.set_trace()
        out = self.linear(out)
        out = out.transpose(1,2).contiguous()
        out = self.relu(self.bn(out))
        out = out.transpose(1,2).contiguous()
        out = self.linear2(out)
        out = out.transpose(0,1).contiguous()
        out = F.log_softmax(out, dim=2)

        return out, out_lens

if __name__ == "__main__":
    a = torch.randn((158, 8, 40))
    model = BiLSTM(40, 256, 47)
    output = model(a)
    print(output.shape)
        
        
