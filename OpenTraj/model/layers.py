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
    def __init__(self, d_model, add_feats=[],add_embeds=[],dis_feats=[], num_embeds=[], road_d=[], con_feats=[],
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

        dim_u=road_d[0]
        dim_s1=road_d[1]
        dim_s2=road_d[2]
        dim_s3=road_d[3]
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
    def __init__(self, d_model, add_feats=[],add_embeds=[],dis_feats=[], num_embeds=[], road_d=[], con_feats=[], kernel_size=3,
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

        dim_u=road_d[0]
        dim_s1=road_d[1]
        dim_s2=road_d[2]
        dim_s3=road_d[3]
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
 
class MutualSemanticDistillationProjector(nn.Module):
    """
    Mutual Semantic Distillation
    """
    def __init__(self, emb_size, d_model, meaningful_anchors, virtual_anchors, n_heads,
                 dropout=0.1, save_attn_map=False, config=None):
        super().__init__()
        
        self.emb_size = emb_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.meaningful_anchors = meaningful_anchors
        self.virtual_anchors = virtual_anchors
        self.num_anchors = meaningful_anchors.size(0) + virtual_anchors.size(0)
        
        self.normalize_V = False  
        self.normalize_F = True   
        
        self.W1 = nn.Parameter(torch.randn(emb_size, emb_size))  
        self.W2 = nn.Parameter(torch.randn(emb_size, emb_size))  
        self.W3 = nn.Parameter(torch.randn(emb_size, emb_size))  
        
        self.W1_1 = nn.Parameter(torch.randn(emb_size, emb_size))  
        self.W2_1 = nn.Parameter(torch.randn(emb_size, emb_size))  
        self.W3_1 = nn.Parameter(torch.randn(emb_size, emb_size))  
        #self.W_att = nn.Parameter(torch.randn(emb_size, emb_size))
        
        self.fusion_gate = nn.Sequential(
            nn.Linear(emb_size * 2, emb_size),
            nn.Sigmoid()
        )
        
        
        self.layer_norm = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(dropout)
        
        self.save_attn_map = save_attn_map
        self.batch_count = 0
    def get_anchors(self):
        anchors = torch.cat([self.meaningful_anchors, self.virtual_anchors], dim=0)
        if self.normalize_V:
            anchors = F.normalize(anchors, dim=1)
        return anchors

    def normalize_features(self, x):
        if self.normalize_F:
            return F.normalize(x, dim=1)
        return x

    def traj_to_semantics_attention(self, x):
        """
        Trajectory → Semantics Attention Subnet
        input: x (B, L, E) - Trajectory
        output: phi (B, K) , beta (B, K, L) , M (B, K, E)
        """
        B, L, E = x.shape
        anchors = self.get_anchors()  # (K, E)
        K = anchors.size(0)
        x_norm = self.normalize_features(x)
        
        S = torch.einsum('ke,ef,ble->bkl', anchors, self.W1, x_norm)  # (B, K, L)
        
        A = torch.einsum('ke,ef,ble->bkl', anchors, self.W2, x_norm)
        A = F.softmax(A, dim=-1) 
        
        M = torch.einsum('bkl,ble->bke', A, x_norm)  # (B, K, E)
        
        phi = torch.einsum('ke,ef,bke->bk', anchors, self.W3, M)
        phi = torch.sigmoid(phi) 
        
        if self.save_attn_map:
            np.save(f'epoch0/traj_to_sem_attn_{self.batch_count}.npy', 
                   A.detach().cpu().numpy())
        #print(phi)
        return phi, A, M

    def semantics_to_traj_attention(self, x):
        """
        Semantics → Trajectory Attention Subnet
        input: x (B, L, E) 
        output: omega (B, K) , T (B, L, K), S (B, L, E) 
        """
        B, L, E = x.shape
        anchors = self.get_anchors()  # (K, E)
        K = anchors.size(0)
        
        x_norm = self.normalize_features(x)
        
        S = torch.einsum('ble,ef,kf->blk', x_norm, self.W1_1, anchors)  # (B, L, K)
        
        T = torch.einsum('ble,ef,kf->blk', x_norm, self.W2_1, anchors)
        T = F.softmax(T, dim=-1) 
        
        v_a = torch.einsum('blk,ke->ble', T, anchors)  # (B, L, E)
        
        omega = torch.einsum('ble,ef,ble->bl', x_norm, self.W3_1, v_a)  # (B, L)

        
        #omega_k = torch.matmul(omega, self.mapping_matrix)  # (B, L) × (L, K) = (B, K) 
        omega_k = torch.einsum('bl,blk->bk', omega, T)
        omega_k = torch.sigmoid(omega_k)
        
        if self.save_attn_map:
            np.save(f'epoch0/sem_to_traj_attn_{self.batch_count}.npy', 
                   T.detach().cpu().numpy())
        #print(omega_k)
        return omega_k, T, v_a


    def semantic_fusion(self, x, phi, omega_k, anchors, A, T):
        """Pattern-Aware Fusion"""
        B, L, E = x.shape
        K = anchors.size(0)
        
        #combined_confidence = torch.sigmoid((phi + omega_k) / 2)
        combined_confidence = (phi + omega_k) / 2
        
        weighted_anchors = torch.einsum('bk,ke->bke', combined_confidence, anchors)  # (B, K, E)
        
        semantic_context = torch.einsum('blk, bke -> ble', T, weighted_anchors) # (B, L, E)

        gate = self.fusion_gate(torch.cat([x, semantic_context], dim=-1))
        fused_output = gate * x + (1 - gate) * semantic_context
        
        return fused_output


    def forward(self, x):
        B, L, E = x.shape
        anchors = self.get_anchors()
        
        phi, A_traj_to_sem, M = self.traj_to_semantics_attention(x)
        
        omega_k, T_sem_to_traj, v_a = self.semantics_to_traj_attention(x)
        
        fused = self.semantic_fusion(x, phi, omega_k, anchors,A_traj_to_sem, T_sem_to_traj)
        
        output = self.layer_norm(x + self.dropout(fused))
        
        self.batch_count += 1
        
        #return output
        return {
            'output': output,
            'phi': phi,  
            'omega_k': omega_k,  
            'A_traj_to_sem': A_traj_to_sem,  
            'T_sem_to_traj': T_sem_to_traj  
        }


class PatternSemanticProjector(nn.Module):
    def __init__(self, emb_size, d_model, meaningful_anchors, virtual_anchors, n_heads,
                 dropout=0.1, save_attn_map=False, config=None) -> None:
        super().__init__()

        self.mhca = MutualSemanticDistillationProjector(
            emb_size, d_model, meaningful_anchors, virtual_anchors, n_heads,
            dropout=dropout, save_attn_map=save_attn_map, config=config
        )
        
      
        self.ffn = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_size, emb_size)
        )
        self.layer_norm = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        
        mhca_output = self.mhca(x)
        
        
        ff_output = self.ffn(mhca_output['output'])
        output = self.layer_norm(x + self.dropout(ff_output))
         
        
        #return output
        return {
            'output': output,
            'phi': mhca_output['phi'],
            'omega_k': mhca_output['omega_k'],
            'A_traj_to_sem': mhca_output['A_traj_to_sem'],
            'T_sem_to_traj': mhca_output['T_sem_to_traj']
        }
