# coding=utf-8
# author=maziqing
# email=maziqing.mzq@alibaba-inc.com


import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed_PointFormer import DataEmbedding
from layers.Causal_Conv import CausalConv
from layers.Multi_Correlation import AutoCorrelation, AutoCorrelationLayer, CrossCorrelation, CrossCorrelationLayer, \
    MultiCorrelation
from layers.Corrformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, \
    my_Layernorm, series_decomp



class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.enc_embedding = DataEmbedding

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        pass
