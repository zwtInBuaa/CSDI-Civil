import torch
from torch import nn

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True)
        self.mlp = nn.Conv1d(in_channels=2*hidden_size, out_channels=input_size, kernel_size=1)

    def forward(self, x):
        # x: [seq_len, batch_size, input_size]
        output, _ = self.lstm(x)  # output: [seq_len, batch_size, 2*hidden_size]
        output = self.mlp(output.permute(1,2,0))  # output: [batch_size, input_size, seq_len]
        # reshape output to [seq_len, batch_size, input_size]
        output = output.permute(2,0,1)
        return output
