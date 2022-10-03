import torch
from torch import nn
import numpy as np
from parameters import N_WORDS_ENCODER, N_WORDS_DECODER, D_MODEL, N, H, P_DROPOUT
import math

D_K = D_Q = D_MODEL // H
D_V = D_MODEL // H


# Scaled dot-product layer
def scaled_dot_product(queries, keys, values, mask=False):
    attention = queries @ torch.transpose(keys, -2, -1)
    attention /= np.sqrt(D_K)
    if mask:
        attention = torch.tril(attention)
    attention = nn.functional.softmax(attention, dim=-1)
    new_values = attention @ values
    return new_values


# Multi head attention layer
class MultiHeadAttention(nn.Module):
    def __init__(self, mask=False):
        super(MultiHeadAttention, self).__init__()
        self.V = nn.Linear(D_MODEL, D_V * H)
        self.Q = nn.Linear(D_MODEL, D_Q * H)
        self.K = nn.Linear(D_MODEL, D_K * H)
        self.linear = nn.Linear(D_V * H, D_MODEL)

    def forward(self, xv, xk, xq):
        v = self.V(xv).view(xv.shape[0], xv.shape[1], H, D_V)
        k = self.K(xk).view(xk.shape[0], xk.shape[1], H, D_K)
        q = self.Q(xq).view(xq.shape[0], xq.shape[1], H, D_Q)
        v = torch.transpose(v, -2, -3)
        k = torch.transpose(k, -2, -3)
        q = torch.transpose(q, -2, -3)
        new_values = scaled_dot_product(queries=q, keys=k, values=v)
        new_values = new_values.view(new_values.shape[0], new_values.shape[2], D_V * H)
        output = self.linear(new_values)
        return output


# Encoder block
class EncoderBlock(nn.Module):
    def __init__(self):
        super(EncoderBlock, self).__init__()
        self.MHA = MultiHeadAttention()
        self.normalization = nn.LayerNorm(D_MODEL)
        self.linear1 = nn.Linear(D_MODEL, D_MODEL * 4)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(D_MODEL * 4, D_MODEL)
        self.dropout1 = nn.Dropout(P_DROPOUT)
        self.dropout2 = nn.Dropout(P_DROPOUT)

    def forward(self, x):
        y = x + self.dropout1(self.MHA(x, x, x))
        y = self.normalization(y)
        z = self.linear1(y)
        z = self.activation(z)
        z = self.linear2(z)
        z = y + self.dropout2(z)
        z = self.normalization(z)
        return z


# Decoder block
class DecoderBlock(nn.Module):
    def __init__(self):
        super(DecoderBlock, self).__init__()
        self.mmha = MultiHeadAttention(mask=True)
        self.mha = MultiHeadAttention()
        self.normalization = nn.LayerNorm(D_MODEL)
        self.linear1 = nn.Linear(D_MODEL, D_MODEL * 4)
        self.linear2 = nn.Linear(D_MODEL * 4, D_MODEL)
        self.activation = nn.ReLU()
        self.dropout1 = nn.Dropout(P_DROPOUT)
        self.dropout2 = nn.Dropout(P_DROPOUT)
        self.dropout3 = nn.Dropout(P_DROPOUT)

    # x - input from previous decoder layer
    # y - output from the encoder layer
    def forward(self, x, y):
        x = x + self.dropout1(self.mmha(x, x, x))
        x = self.normalization(x)
        x = x + self.dropout2(self.mha(y, y, x))
        x = self.normalization(x)
        z = self.linear1(x)
        z = self.activation(z)
        z = self.linear2(z)
        z = x + self.dropout3(z)
        z = self.normalization(z)
        return z


class PositionalEncoding(nn.Module):

    def __init__(self, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.pe = torch.zeros(max_len, D_MODEL)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D_MODEL, 2).float() * (-math.log(10000.0) / D_MODEL))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.dropout = nn.Dropout(P_DROPOUT)

    def forward(self, x):
        return self.dropout(x + self.pe[:x.size(1), :])


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=N_WORDS_ENCODER, embedding_dim=D_MODEL)
        self.positionalEnc = PositionalEncoding()
        self.encoderBlocks = nn.ModuleList([EncoderBlock() for _ in range(N)])

    def forward(self, inp):
        out = self.embedding(inp)
        out = self.positionalEnc(out)
        for eb in self.encoderBlocks:
            out = eb(out)
        return out


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=N_WORDS_DECODER + 1, embedding_dim=D_MODEL, padding_idx=N_WORDS_DECODER)
        self.positionalEnc = PositionalEncoding()
        self.decoderBlocks = nn.ModuleList([DecoderBlock() for _ in range(N)])
        self.linear = nn.Linear(D_MODEL, N_WORDS_DECODER)

    def forward(self, inp, encoder_output):
        out = self.embedding(inp)
        out = self.positionalEnc(out)
        for db in self.decoderBlocks:
            out = db(out, encoder_output)
        out = self.linear(out)
        out = nn.functional.softmax(out, dim=-1)
        return out


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, encoder_input, decoder_input):
        encoder_output = self.encoder(encoder_input)
        output = self.decoder(decoder_input, encoder_output)
        return output
