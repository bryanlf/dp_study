import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from positional_encoding import SinusoidalPositionalEncoder, LearnablePositionalEncoder


class Transformer(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.window_len = args.window_len
        self.target_len = args.target_len
        self.num_encoder_layers = args.num_encoder_layers
        self.num_decoder_layers = args.num_decoder_layers
        self.input_size = args.input_size
        self.output_size = args.out_size
        self.d_model = args.d_model
        self.num_heads = args.num_heads
        self.feedforward_dim = args.feedforward
        self.dropout = args.dropout
        self.positional_encoding = args.positional_encoding
        # 位置编码
        if self.positional_encoding == "sinusoidal":
            self.positional_encoder = SinusoidalPositionalEncoder(seq_len=self.window_len, d_model=self.d_model,
                                                                  dropout=self.dropout)
            self.positional_decoder = SinusoidalPositionalEncoder(seq_len=self.target_len, d_model=self.d_model,
                                                                  dropout=self.dropout)
        elif self.positional_encoding == "learnable":
            self.positional_encoder = LearnablePositionalEncoder(seq_len=self.window_len, d_model=self.d_model,
                                                                 dropout=self.dropout)
            self.positional_decoder = LearnablePositionalEncoder(seq_len=self.target_len, d_model=self.d_model,
                                                                 dropout=self.dropout)
        else:
            raise Exception("Positional encoding type not recognized: use 'sinusoidal' or 'learnable'.")
        # 编码层
        self.encoder_input_layer = nn.Linear(self.input_size, self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.num_heads,
                                                   dim_feedforward=self.feedforward_dim,
                                                   dropout=self.dropout,
                                                   batch_first=True)
        # 表示n个编码器层的堆叠
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=self.num_encoder_layers)

        self.decoder_input_layer = nn.Linear(self.output_size, self.d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.d_model, nhead=self.num_heads,
                                                   dim_feedforward=self.feedforward_dim,
                                                   dropout=self.dropout,
                                                   batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=self.num_decoder_layers)

        self.output_layer = nn.Linear(self.d_model, self.output_size)

        self.init_weights()

    #权重初始化
    def init_weights(self):
        initrange = 0.1
        self.encoder_input_layer.weight.data.uniform_(-initrange, initrange)
        self.decoder_input_layer.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.zero_()
        self.output_layer.weight.data.uniform_(-initrange, initrange)

    def generate_mask(self, dim1, dim2):
        return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)
        # torch.triu(diagonal=1)函数返回主对角线上的元素，其余元素为0,这里相当与创建了一个对角线为1的矩阵

    def forward(self, src, trg, memory_mask=None, trg_mask=None):
        src = self.encoder_input_layer(src)
        src = self.positional_encoder(src)
        encoder_output = self.encoder(src)
        trg = self.decoder_input_layer(trg)
        trg = self.positional_decoder(trg)

        trc = trg.permute(1, 0, 2).contiguous()
        #print(trc.shape)
        if memory_mask is None:
            memory_mask = self.generate_mask(self.target_len, self.window_len).to(src.device)
        if trg_mask is None:
            trg_mask = self.generate_mask(self.target_len, self.target_len).to(src.device)
        decoder_output = self.decoder(trc, encoder_output)
        #print(decoder_output.shape)
        output = self.output_layer(decoder_output)
        #print(output.shape)

        return output
