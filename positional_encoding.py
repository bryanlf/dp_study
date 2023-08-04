import torch
import torch.nn as nn
import math


class SinusoidalPositionalEncoder(nn.Module):
    def __init__(self, seq_len, d_model, dropout=0.1):
        super(SinusoidalPositionalEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # d_model表示位置编码的维度，
        position = torch.arange(seq_len).unsqueeze(1)  # pos表位置索引，这里是指seq_len是指时间步长
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(1, seq_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe

        return self.dropout(x)


class LearnablePositionalEncoder(nn.Module):
    def __init__(self, seq_len, d_model, dropout=0.1):
        super(LearnablePositionalEncoder, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        pe = nn.parameter.Parameter(torch.zeros((1, seq_len, d_model)))
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe

        return self.dropout(x)
