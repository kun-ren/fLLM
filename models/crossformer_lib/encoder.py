import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from math import ceil

from models.crossformer_lib.attentation import TwoStageAttentionLayer


class SegMerging(nn.Module):
    '''
    Segment Merging Layer.
    The adjacent `aggregation_level' segments in each dimension will be merged into one segment to
    get representation of a coarser scale
    we set aggregation_level = 2 in our paper
    '''

    def __init__(self, d_model, aggregation_level, norm_layer=nn.LayerNorm):
        super().__init__()
        self.d_model = d_model
        self.aggregation_level = aggregation_level
        self.linear_trans = nn.Linear(aggregation_level * d_model, d_model)
        self.norm = norm_layer(aggregation_level * d_model)

    def forward(self, x):
        """
        x: B, ts_d, L, d_model
        """

        batch_size, ts_d, seg_num, d_model = x.shape
        pad_num = seg_num % self.aggregation_level
        if pad_num != 0:
            pad_num = self.aggregation_level - pad_num
            x = torch.cat((x, x[:, :, -pad_num:, :]), dim=-2)

        seg_to_merge = []
        for i in range(self.aggregation_level):
            seg_to_merge.append(x[:, :, i::self.aggregation_level, :])
        x = torch.cat(seg_to_merge, -1)  # [B, ts_d, seg_num/aggregation_level, aggregation_level*d_model]

        x = self.norm(x)
        x = self.linear_trans(x)

        return x


class scale_block(nn.Module):
    '''
    We can use one segment merging layer followed by multiple TSA layers in each scale
    the parameter `depth' determines the number of TSA layers used in each scale
    We set depth = 1 in the paper
    '''

    def __init__(self, aggregation_level, d_model, n_heads, d_ff, num_TSA_layer, dropout, seg_num=10, factor=10,
                 router=False):
        super(scale_block, self).__init__()

        if (aggregation_level > 1):
            self.merge_layer = SegMerging(d_model, aggregation_level, nn.LayerNorm)
        else:
            self.merge_layer = None

        self.tsa_layers = nn.ModuleList()

        for i in range(num_TSA_layer):
            self.tsa_layers.append(TwoStageAttentionLayer(seg_num, factor, d_model, n_heads, d_ff, dropout, router=router))

    def forward(self, x):
        _, ts_dim, _, _ = x.shape

        if self.merge_layer is not None:
            x = self.merge_layer(x)

        for layer in self.tsa_layers:
            x = layer(x)

        return x


class Encoder(nn.Module):
    '''
    The Encoder of Crossformer.

        :param
        d_ff: The dimension of the Feed-Forward network (the linear layers after attention).
        Usually, this is set to 2 or 4 times the d_model to allow for non-linear data expansion and compression.



    '''

    def __init__(self, num_encoder_layer, aggregation_level, d_model, n_heads, d_ff, num_tsa_layer, dropout,
                 total_seg_num=10, factor=10, router=False):
        super(Encoder, self).__init__()
        self.encode_blocks = nn.ModuleList()

        self.encode_blocks.append(scale_block(1, d_model, n_heads, d_ff, num_tsa_layer, dropout,
                                              total_seg_num, factor, router=router))
        for i in range(1, num_encoder_layer):
            self.encode_blocks.append(scale_block(aggregation_level, d_model, n_heads, d_ff, num_tsa_layer, dropout,
                                                  ceil(total_seg_num / aggregation_level ** i), factor, router=router))

    def forward(self, x):
        encode_x = []
        encode_x.append(x)

        for block in self.encode_blocks:
            x = block(x)
            encode_x.append(x)

        return encode_x
