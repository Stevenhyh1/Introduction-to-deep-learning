import numpy as np

import torch
import torch.nn as nn
import torch.nn.utils as utils
from torch.autograd import Variable

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x


class PyramidLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PyramidLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.pblstm = nn.LSTM(self.input_dim, self.hidden_dim, num_layers=1, bidirectional=True)
        
    def forward(self, x):
        """
        :param x: Input sequence: (T, B, D)
        :return: output: Output sequence: (T/2, B, D*2)
        """

        x, lengths = utils.rnn.pad_packed_sequence(x)
        batch_size = x.size(1)

        if x.size(0) % 2 is not 0:
            padding = torch.zeros(1, batch_size, self.hidden_dim*2).to(DEVICE)
            x = torch.cat((x, padding), dim=0)
            # x = x[:-1]
        
        new_length = int(x.size(0)/2)
        x = x.transpose(0, 1)  # (B, T, D)
        x = x.reshape(batch_size, new_length, self.hidden_dim*4)  # (B, T/2, D*2)
        x = x.transpose(0, 1)  # (T/2, B, D*2)

        for i, sample_len in enumerate(lengths):
            if sample_len % 2 == 0:
                sample_len = int(sample_len/2)
            else:
                sample_len = int(sample_len/2) + 1
            lengths[i] = sample_len
        # lengths //= 2

        x = utils.rnn.pack_padded_sequence(x, lengths, enforce_sorted=False)

        output, _ = self.pblstm(x)

        return output, lengths  


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, value_size, key_size, pyramid_layer, bidirectional=True):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dim
        self.pyramid_layer = pyramid_layer
        self.value_size = value_size
        self.key_size = key_size
        self.bidirectional = bidirectional
        
        self.blstm = nn.LSTM(self.input_dim, self.hidden_dims, num_layers=1, bidirectional=self.bidirectional)
        self.lockdrop = LockedDropout()

        self.pblstm = []
        for i in range(self.pyramid_layer):
            self.pblstm.append(PyramidLSTM(self.hidden_dims*4, self.hidden_dims))
        self.pblstm = nn.ModuleList(self.pblstm)

        self.value_network = nn.Linear(self.hidden_dims*2, self.value_size)
        self.key_network = nn.Linear(self.hidden_dims*2, self.key_size)

    def forward(self, x, lengths):
        """
        :param x: Padded sequence of dimension (T, B, D), T is the length of the longest sequence
        :param lengths: List of sequence lengths
        :return: keys: keys for attention layer
                 values: values for attention layer
        """
        # import pdb; pdb.set_trace()
        packed_x = utils.rnn.pack_padded_sequence(x, lengths, enforce_sorted=False)
        outputs, _ = self.blstm(packed_x)
        
        for i, pblstm_layer in enumerate(self.pblstm):
            outputs, lengths = pblstm_layer(outputs)
            outputs, out_lens = utils.rnn.pad_packed_sequence(outputs)
            outputs = self.lockdrop(outputs, 0.2)
            if i != self.pyramid_layer-1:
                outputs = utils.rnn.pack_padded_sequence(outputs, out_lens, enforce_sorted=False)
        
        # import pdb; pdb.set_trace()
        linear_input = outputs
        keys = self.key_network(linear_input)
        values = self.value_network(linear_input)

        return keys, values, lengths


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

        '''
        Attention is calculated using key, value and query from Encoder and decoder.
        Below are the set of operations you need to perform for computing attention:
            bmm: batch matrix multiplication
            energy = bmm(key, query) 
            attention = softmax(energy)
            context = bmm(attention, value)
        '''

    def forward(self, query, key, value, lens):
        """
        :param key:  (Max_len, B, key_size), Key projection of encoder
        :param value: (Max_len, B, value_size), Value projection of encoder
        :param query: (B, hidden), current state of the decoder
        :param lens: list of sequence length of the encoder
        :return: attention_context, attention_mask
        """

        assert query.size(1) == key.size(2), 'Key dimension not matching hidden states dimension'
        assert query.size(1) == value.size(2), 'Key dimension not matching hidden states dimension'
        max_len = key.size(0)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
        energy = torch.bmm(key, query.unsqueeze(2)) 
        attention_mask = torch.arange(max_len).unsqueeze(0) >= lens.unsqueeze(1)
        attention_mask = attention_mask.unsqueeze(2).to(DEVICE) 
        energy.masked_fill_(attention_mask, -1e9)
        attention = nn.functional.softmax(energy, dim=1)
        attention_context = torch.bmm(attention.transpose(2,1), value).squeeze(1)

        return attention_context, attention


class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, value_size, key_size, use_attention=True):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.value_size = value_size
        self.key_size = key_size
        self.use_attention = use_attention

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.lstm1 = nn.LSTMCell(input_size=hidden_dim+value_size, hidden_size=2*hidden_dim)
        self.lstm2 = nn.LSTMCell(input_size=2*hidden_dim, hidden_size=key_size)

        if self.use_attention:
            self.attention = Attention()

        self.linear = nn.Linear(key_size+value_size,vocab_size)

    def forward(self, keys, values, lens, epoch, text=None, istrain=True):

        """
        :param keys: (max_len, B, key_size) output of the encoder keys projection
        :param values: (max_len, B, value_size) output of the encoder value projection
        :param text: (B, max_len) batch input of text
        :param lens: (B, ) lengths of the batch input sequences
        :param istrain: train or evaluation mode
        :return: predictions: character prediction probability
        """

        batch_size = keys.size(1)

        if istrain:
            max_len =  text.size(1)
            embeddings = self.embedding(text)
        else:
            max_len = 250

        predictions = []
        hidden_states = [None, None]
        prediction = torch.zeros(batch_size,1).to(DEVICE)
        
        if self.use_attention:
            context = torch.zeros(batch_size,keys.size(2)).to(DEVICE)

        for i in range(max_len):

            teacher_forcing = np.random.uniform(0,1)
            if epoch < 20:
                p_forcing = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
                             0.12, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
                cur_p = p_forcing[epoch]
            else:
                cur_p = 0.5
            
            if istrain and teacher_forcing > cur_p:
                embedding_letter = embeddings[:, i, :]
            else:
                embedding_letter = self.embedding(prediction.argmax(dim=-1))

            if self.use_attention:
                input1 = torch.cat((embedding_letter, context), dim=1)
            else:
                input1 = torch.cat((embedding_letter, values[i, :, :]), dim=1)            

            hidden_states[0] = self.lstm1(input1,hidden_states[0])

            input2 = hidden_states[0][0]
            hidden_states[1] = self.lstm2(input2,hidden_states[1])
            
            lstm_outputs = hidden_states[1][0]
            if self.use_attention:
                context, mask = self.attention(lstm_outputs, keys, values, lens)

            linear_input = torch.cat([lstm_outputs, context], dim=1)
            prediction = self.linear(linear_input)

            # if istrain:
            predictions.append(prediction.unsqueeze(1))
            # else:
            #     predictions.append(predictions.argmax(dim=-1).unsqueeze)

        return torch.cat(predictions, dim=1)


class Seq2Seq(nn.Module):
    def __init__(self, input_dim, vocab_size, hidden_dim, value_size, key_size, pyramidlayers, use_attention=True):
        super(Seq2Seq,self).__init__()
        
        self.input_dim = input_dim
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.value_size = value_size
        self.key_size = key_size
        self.pyramidlayers = pyramidlayers
        self.useattention = use_attention

        self.encoder = Encoder(self.input_dim, self.hidden_dim, self.value_size, self.key_size, self.pyramidlayers)
        self.decoder = Decoder(self.vocab_size, self.hidden_dim, self.value_size, self.key_size, self.useattention)
        

    def forward(self, speech_input, speech_lens, epoch=0, text_input=None, istrain=True):
            
        key, value, lens = self.encoder(speech_input, speech_lens)
        if istrain:
            predictions = self.decoder(keys=key, values=value, lens=lens, epoch=epoch, text=text_input, istrain=True)
        else:
            predictions = self.decoder(keys=key, values=value, lens=lens, epoch=None, text=None, istrain=False)
        return predictions


# if __name__ == "__main__":
#     T = 200
#     B = 8
#     D = 256
#     torch.manual_seed(11785)
#     letter_list = ['<pad>', "'", '+', '-', '.', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
#                    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '_', ' ', '<sos>', '<eos>']
#     inputs = torch.randn((T, B, D), requires_grad=True)
#     lengths = torch.randint(low=50, high=100, size=(B, )).tolist()
#     print(lengths)
#     max_len = max(lengths)
#     texts = torch.zeros((B, max_len), dtype=torch.long)
#     for i in range(B):
#         text = torch.randint(low=0, high=len(letter_list)-1, size=(1, lengths[i]), dtype=torch.long)
#         texts[i][:lengths[i]] = text

    # Test PyramidLSTM
    # model = PyramidLSTM(D, 512)
    # outputs = model(inputs)
    # print(inputs.size())
    # print(outputs.size())

    # Test Encoder
    # model = Encoder(D, 256, 3, 128, 128)
    # keys, values, lens = model(inputs, lengths)
    # print(lengths)
    # print(inputs.size())
    # print(keys.size())
    # print(values.size())

    # Test Attention
    # model = Attention()
    # queries = torch.randn(B, 128)
    # context, mask = model(keys, values, queries, lens)
    # print(context.size())
    # print(mask.size())

    # Test Decoder
    # decoder = Decoder(vocab_size=len(letter_list), hidden_dim=D,
    #                   value_size=128, key_size=128, use_attention=True)
    # predictions = decoder(keys=keys, values=values, lens=lens, text=texts, istrain=True)
    # print(predictions.size())

    # Test Seq2Seq
    # model = Seq2Seq(input_dim=D, vocab_size=len(letter_list), hidden_dim=D,
    #                 value_size=128, key_size=128, pyramidlayers=3, useattention=True)
    # model.to(DEVICE)
    # inputs = inputs.to(DEVICE)
    # texts = texts.to(DEVICE)
    # predictions = model(speech_input=inputs, speech_lens=lengths, text_input=texts, istrain=True)
    # print(predictions.size())