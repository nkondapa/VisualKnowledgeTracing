import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

'''
from - https://github.com/arshadshk/SAINT-pytorch/blob/main/saint.py
'''

class FeedForwardBlock(nn.Module):
    """
    out =  Relu( M_out*w1 + b1) *w2 + b2
    """

    def __init__(self, dim_ff):
        super().__init__()
        self.layer1 = nn.Linear(in_features=dim_ff, out_features=dim_ff)
        self.layer2 = nn.Linear(in_features=dim_ff, out_features=dim_ff)

    def forward(self, ffn_in):
        return self.layer2(F.relu(self.layer1(ffn_in)))


class EncoderBlock(nn.Module):
    """
    M = SkipConct(Multihead(LayerNorm(Qin;Kin;Vin)))
    O = SkipConct(FFN(LayerNorm(M)))
    """

    def __init__(self, encoder_in_dim, dim_model, heads_en, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.dim_model = dim_model
        self.encoder_in_dim = encoder_in_dim

        self.init_transform = nn.Linear(encoder_in_dim, dim_model)
        self.embd_pos = nn.Embedding(seq_len, embedding_dim=dim_model)  # positional embedding
        self.multi_en = nn.MultiheadAttention(embed_dim=dim_model,
                                              num_heads=heads_en, )  # multihead attention    ## todo add dropout, LayerNORM
        self.ffn_en = FeedForwardBlock(dim_model)  # feedforward block     ## todo dropout, LayerNorm
        self.layer_norm1 = nn.LayerNorm(dim_model)
        self.layer_norm2 = nn.LayerNorm(dim_model)

    def forward(self, vec, first_block=True):

        if first_block:
            vec = self.init_transform(vec)

        pos_emb = get_pos(self.seq_len).to(vec.device)
        if vec.shape[1] > pos_emb.shape[1]:
            pos_emb = torch.cat([pos_emb, pos_emb[:, [-1]]], dim=-1)
        pos_emb = self.embd_pos(pos_emb)
        out = vec + pos_emb  # Applying positional embedding

        out = out.permute(1, 0, 2)  # (n,b,d)  # print('pre multi', out.shape )

        # Multihead attention
        n, _, _ = out.shape
        out = self.layer_norm1(out)  # Layer norm
        skip_out = out
        out, attn_wt = self.multi_en(out, out, out,
                                     attn_mask=get_mask(seq_len=n, device=vec.device))  # attention mask upper triangular
        out = out + skip_out  # skip connection

        # feed forward
        out = out.permute(1, 0, 2)  # (b,n,d)
        out = self.layer_norm2(out)  # Layer norm
        skip_out = out
        out = self.ffn_en(out)
        out = out + skip_out  # skip connection

        return out


class DecoderBlock(nn.Module):
    """
    M1 = SkipConct(Multihead(LayerNorm(Qin;Kin;Vin)))
    M2 = SkipConct(Multihead(LayerNorm(M1;O;O)))
    L = SkipConct(FFN(LayerNorm(M2)))
    """

    def __init__(self, decoder_in_dim, dim_model, heads_de, seq_len):
        super().__init__()
        self.seq_len = seq_len

        self.init_transform = nn.Linear(decoder_in_dim, dim_model)
        self.embd_pos = nn.Embedding(seq_len, embedding_dim=dim_model)  # positional embedding
        self.multi_de1 = nn.MultiheadAttention(embed_dim=dim_model,
                                               num_heads=heads_de)  # M1 multihead for interaction embedding as q k v
        self.multi_de2 = nn.MultiheadAttention(embed_dim=dim_model,
                                               num_heads=heads_de)  # M2 multihead for M1 out, encoder out, encoder out as q k v
        self.ffn_en = FeedForwardBlock(dim_model)  # feed forward layer

        self.layer_norm1 = nn.LayerNorm(dim_model)
        self.layer_norm2 = nn.LayerNorm(dim_model)
        self.layer_norm3 = nn.LayerNorm(dim_model)

    def forward(self, vec, encoder_out, first_block=True):

        if first_block:
            vec = self.init_transform(vec)

        pos_emb = get_pos(self.seq_len).to(vec.device)
        if vec.shape[1] > pos_emb.shape[1]:
            pos_emb = torch.cat([pos_emb, pos_emb[:, [-1]]], dim=-1)
        in_pos = self.embd_pos(pos_emb)
        out = vec + in_pos  # Applying positional embedding

        out = out.permute(1, 0, 2)  # (n,b,d)# print('pre multi', out.shape )
        n, _, _ = out.shape

        # Multihead attention M1                                     ## todo verify if E to passed as q,k,v
        out = self.layer_norm1(out)
        skip_out = out
        out, attn_wt = self.multi_de1(out, out, out,
                                      attn_mask=get_mask(seq_len=n, device=vec.device))  # attention mask upper triangular
        out = skip_out + out  # skip connection

        # Multihead attention M2                                     ## todo verify if E to passed as q,k,v
        en_out = encoder_out.permute(1, 0, 2)  # (b,n,d)-->(n,b,d)
        en_out = self.layer_norm2(en_out)
        skip_out = out
        out, attn_wt = self.multi_de2(out, en_out, en_out,
                                      attn_mask=get_mask(seq_len=n, device=vec.device))  # attention mask upper triangular
        out = out + skip_out

        # feed forward
        out = out.permute(1, 0, 2)  # (b,n,d)
        out = self.layer_norm3(out)  # Layer norm
        skip_out = out
        out = self.ffn_en(out)
        out = out + skip_out  # skip connection

        return out


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_mask(seq_len, device=None):
    return torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1).astype('bool')).to(device)


def get_pos(seq_len):
    # use sine positional embeddinds
    return torch.arange(seq_len).unsqueeze(0)


class saint(nn.Module):
    def __init__(self, dim_model, num_en, num_de, heads_en, total_ex, total_cat, total_in, heads_de, seq_len):
        super().__init__()

        self.num_en = num_en
        self.num_de = num_de

        self.encoder = get_clones(EncoderBlock(dim_model, heads_en, total_ex, total_cat, seq_len), num_en)
        self.decoder = get_clones(DecoderBlock(dim_model, total_in, heads_de, seq_len), num_de)

        self.out = nn.Linear(in_features=dim_model, out_features=1)

    def forward(self, in_ex, in_cat, in_in):

        ## pass through each of the encoder blocks in sequence
        first_block = True
        for x in range(self.num_en):
            if x >= 1:
                first_block = False
            in_ex = self.encoder[x](in_ex, in_cat, first_block=first_block)
            in_cat = in_ex  # passing same output as q,k,v to next encoder block

        ## pass through each decoder blocks in sequence
        first_block = True
        for x in range(self.num_de):
            if x >= 1:
                first_block = False
            in_in = self.decoder[x](in_in, en_out=in_ex, first_block=first_block)

        ## Output layer
        in_in = torch.sigmoid(self.out(in_in))
        return in_in


