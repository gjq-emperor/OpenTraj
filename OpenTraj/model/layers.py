import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class ContinuousEncoding(nn.Module):
    """
    A type of trigonometric encoding for encode continuous values into distance-sensitive vectors.
    """

    def __init__(self, embed_size):
        super().__init__()
        self.omega = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, embed_size))).float(),
                                  requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(embed_size).float(), requires_grad=True)
        self.div_term = math.sqrt(1. / embed_size)

    def forward(self, x):
        """
        :param x: input sequence for encoding, (batch_size, seq_len)
        :return: encoded sequence, shape (batch_size, seq_len, embed_size)
        """
        encode = x.unsqueeze(-1) * self.omega.reshape(1, 1, -1) + self.bias.reshape(1, 1, -1)
        encode = torch.cos(encode)
        return self.div_term * encode
    

class PositionalEncoding(nn.Module):
    """
    A type of trigonometric encoding for indicating items' positions in sequences.
    """

    def __init__(self, embed_size, max_len):
        super().__init__()

        pe = torch.zeros(max_len, embed_size).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_size, 2).float() * -(math.log(10000.0) / embed_size)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, position_ids=None):
        """
        Args:
            x: (B, T, d_model)
            position_ids: (B, T) or None

        Returns:
            (1, T, d_model) / (B, T, d_model)
        """
        if position_ids is None:
            return self.pe[:, :x.size(1)]
        else:
            batch_size, seq_len = position_ids.shape
            pe = self.pe[:, :seq_len, :]  # (1, T, d_model)
            pe = pe.expand((position_ids.shape[0], -1, -1))  # (B, T, d_model)
            pe = pe.reshape(-1, self.d_model)  # (B * T, d_model)
            position_ids = position_ids.reshape(-1, 1).squeeze(1)  # (B * T,)
            output_pe = pe[position_ids].reshape(batch_size, seq_len, self.d_model).detach()
            return output_pe
        

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal-based function used for encoding timestamps.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TimeEmbed(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, time):
        return self.time_mlp(time)


class MLP2(nn.Module):
    """
    MLP with two output layers
    """
    def __init__(self, input_size, hidden_size, output_size,
                 dropout, use_selu=False):
        super(MLP2, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc21 = nn.Linear(hidden_size, output_size)
        self.fc22 = nn.Linear(hidden_size, output_size)
        self.nonlinear_f = F.selu if use_selu else F.relu
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        h1 = self.dropout(self.nonlinear_f(self.fc1(x)))
        return self.fc21(h1), self.fc22(h1)


class TrajEmbedding(nn.Module):
    def __init__(self, d_model, add_feats=[],add_embeds=[],dis_feats=[], num_embeds=[], con_feats=[],
                 pre_embed=None, pre_embed_update=False, second_col=None):
        super().__init__()

        self.d_model = d_model
        self.dis_feats = dis_feats
        self.con_feats = con_feats
        self.second_col = second_col

        self.add_feats=add_feats

        if len(dis_feats):
            assert len(dis_feats) == len(num_embeds), \
                'length of num_embeds list should be equal to the number of discrete features.'
            self.dis_embeds = nn.ModuleList([nn.Embedding(num_embed, d_model) for num_embed in num_embeds])
        else:
            self.dis_embeds = None

        if len(con_feats):
            # continuous encoding
            # self.con_embeds = nn.ModuleList([ContinuousEncoding(d_model) for _ in con_feats])
            # linear
            self.con_embeds = nn.Linear(len(con_feats), d_model)
        else:
            self.con_embeds = None

        if pre_embed is not None:
            self.dis_embeds[0].weight = nn.Parameter(torch.from_numpy(pre_embed),
                                                     requires_grad=pre_embed_update)

        if second_col is not None:
            self.time_embed = ContinuousEncoding(d_model)
        num_u=add_embeds[0]
        num_s1=add_embeds[1]
        num_s2=add_embeds[2]
        num_s3=add_embeds[3]

        dim_u=128
        dim_s1=64
        dim_s2=32
        dim_s3=16
        self.embedding_u = nn.Embedding(num_u, dim_u)
        self.embedding_s1 = nn.Embedding(num_s1, dim_s1)
        self.embedding_s2 = nn.Embedding(num_s2, dim_s2)
        self.embedding_s3 = nn.Embedding(num_s3, dim_s3)

        self.f = MLP2(dim_u + dim_s1 + dim_s2 + dim_s3,
                      hidden_size=512, output_size=768, dropout=0.3, use_selu=1)
    

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add(mu)
        else:
            return mu

    def forward(self, x):
        B, L, E_in = x.shape
        u = self.embedding_u(x[...,self.add_feats[0]].long())
        
        s1 = self.embedding_s1(x[...,self.add_feats[1]].long())
        
        s2 = self.embedding_s2(x[...,self.add_feats[2]].long())
        
        s3 = self.embedding_s3(x[...,self.add_feats[3]].long())
        
        cu = torch.cat([u, s1, s2, s3], dim=2)
        mu, logvar = self.f(cu)
        rho=self.reparameterize(mu, logvar)

        h = torch.zeros(B, L, self.d_model).to(x.device)
        if self.dis_embeds is not None:
            for dis_embed, dis_feat in zip(self.dis_embeds, self.dis_feats):
                h += dis_embed(x[..., dis_feat].long())
        # continuous encoding
        # if self.con_embeds is not None:
        #     for con_embed, con_feat in zip(self.con_embeds, self.con_feats):
        #         h += con_embed(x[..., con_feat].float())
        if self.con_embeds is not None:
            h += self.con_embeds(x[..., self.con_feats].float())

        h=h+rho
        if self.second_col is not None:
            h += self.time_embed(x[..., int(self.second_col)])

        return h


class TrajConvEmbedding(nn.Module):
    def __init__(self, d_model, add_feats=[],add_embeds=[],dis_feats=[], num_embeds=[], con_feats=[], kernel_size=3,
                 pre_embed=None, pre_embed_update=False, second_col=None):
        super().__init__()

        self.d_model = d_model
        self.dis_feats = dis_feats
        self.con_feats = con_feats
        self.second_col = second_col

        self.add_feats=add_feats

        # Operates discrete features by look-up table.
        if len(dis_feats):
            assert len(dis_feats) == len(num_embeds), \
                'length of num_embeds list should be equal to the number of discrete features.'
            self.dis_embeds = nn.ModuleList([nn.Embedding(num_embed, d_model) for num_embed in num_embeds])
        else:
            self.dis_embeds = None

        if pre_embed is not None:
            self.dis_embeds[0].weight = nn.Parameter(torch.from_numpy(pre_embed),
                                                     requires_grad=pre_embed_update)

        # Operates continuous features by convolution.
        self.conv = nn.Conv1d(len(con_feats), d_model,
                              kernel_size=kernel_size, padding=(kernel_size - 1)//2)

        # Time embedding
        if second_col is not None:
            self.time_embed = ContinuousEncoding(d_model)
        
        num_u=add_embeds[0]
        num_s1=add_embeds[1]
        num_s2=add_embeds[2]
        num_s3=add_embeds[3]

        dim_u=128
        dim_s1=64
        dim_s2=32
        dim_s3=16
        self.embedding_u = nn.Embedding(num_u, dim_u)
        self.embedding_s1 = nn.Embedding(num_s1, dim_s1)
        self.embedding_s2 = nn.Embedding(num_s2, dim_s2)
        self.embedding_s3 = nn.Embedding(num_s3, dim_s3)

        self.f = MLP2(dim_u + dim_s1 + dim_s2 + dim_s3,
                      hidden_size=512, output_size=768, dropout=0.3, use_selu=1)
    

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add(mu)
        else:
            return mu

    def forward(self, x):
        B, L, E_in = x.shape

        u = self.embedding_u(x[...,self.add_feats[0]].long())
        
        s1 = self.embedding_s1(x[...,self.add_feats[1]].long())
        
        s2 = self.embedding_s2(x[...,self.add_feats[2]].long())
        
        s3 = self.embedding_s3(x[...,self.add_feats[3]].long())
        
        cu = torch.cat([u, s1, s2, s3], dim=2)
        mu, logvar = self.f(cu)
        rho=self.reparameterize(mu, logvar)
        

        h = torch.zeros(B, L, self.d_model).to(x.device)
        if self.dis_embeds is not None:
            for dis_embed, dis_feat in zip(self.dis_embeds, self.dis_feats):
                h += dis_embed(x[..., dis_feat].long())
        h=h+rho
        if self.con_feats is not None:
            h += self.conv(x[..., self.con_feats].transpose(1, 2)).transpose(1, 2)

        if self.second_col is not None:
            h += self.time_embed(x[..., int(self.second_col)])
            
        return h
    

class MSDN(nn.Module):
    def __init__(self, meaningful_anchors, virtual_anchors):
        super(MSDN, self).__init__()
        self.dim_v = 768
        self.meaningful_anchors = meaningful_anchors
        self.virtual_anchors = virtual_anchors
        self.num_anchors = meaningful_anchors.size(0) + virtual_anchors.size(0)
        self.W_q = nn.Linear(self.dim_v, self.dim_v)
        self.W2 = nn.Linear(self.dim_v, self.dim_v)
        self.W4 = nn.Linear(self.dim_v, self.dim_v)

        self.semantic_transform = nn.MultiheadAttention(self.dim_v, num_heads=4, batch_first=True)
        self.gate_mlp = nn.Sequential(
            nn.Linear(2*self.dim_v, 4*self.dim_v),
            nn.GELU(),
            nn.Linear(4*self.dim_v, 2*self.dim_v),
            nn.Tanh()
        )
        self.norm1 = nn.LayerNorm(self.dim_v)
        self.norm2 = nn.LayerNorm(self.dim_v)
        self.res_gate = nn.Parameter(torch.tensor(0.5))
        
        
    def forward(self, x):
        # x: Input trajectory (B, L, d)
        B, L, d = x.shape
        device = x.device
        anchors = torch.concat([self.meaningful_anchors, self.virtual_anchors], dim=0)  # (K, d)
        K = anchors.size(0)
        #################### Trajectory → Semantic attention subnetwork ####################
        Q1 = self.W_q(x)
        att_scores1 = torch.einsum('bld,kd->blk', Q1, anchors) / (d**0.5)
        beta = F.softmax(att_scores1, dim=-1)
        M = torch.einsum('blk,bld->bkd', beta, x)
        M = self.norm1(M) 
        phi_k = torch.einsum('bkd,kd->bk', self.W2(M), anchors)

        #################### Semantic → Trajectory attention subnetwork ####################

        Q2 = self.W_q(x) 
        T = F.softmax(torch.einsum('bld,kd->blk', Q2, anchors), dim=1)
        print("T的形状:",T.shape)
        S = torch.einsum('blk,kd->bld', T, anchors)
        omega_i = torch.einsum('bld,bld->bl', x, self.W4(S))
        
        #################### pattern-aware fusion ####################

        semantic_base = torch.einsum('bk,kd->bkd', phi_k, anchors)
        semantic_feat, _ = self.semantic_transform(x, anchors.expand(B,-1,-1), semantic_base)
        
        trajectory_feat = torch.einsum('bl,bld->bld', omega_i, x)
        
        # 双路门控融合
        combined = torch.cat([semantic_feat, trajectory_feat], dim=-1)
        gate_out = self.gate_mlp(combined)
        gate_sem, gate_traj = gate_out.chunk(2, dim=-1)
        
        fused = gate_sem * semantic_feat + gate_traj * trajectory_feat
        x_out = self.res_gate * x + (1 - self.res_gate) * fused
        return self.norm2(x_out)

class PatternSemanticProjector(nn.Module):
    """ Project movement patterns onto a semantic-rich textual space. """
    
    def __init__(self, emb_size, d_model, meaningful_anchors, virtual_anchors, n_heads,
                 dropout=0.1, save_attn_map=False) -> None:
        super().__init__()
        
        self.mhca = MSDN(meaningful_anchors, virtual_anchors)
        # feedforward layer
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        # self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        return x + self.ffn(self.mhca(x))