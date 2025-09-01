import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from einops import reduce


# SelfAttention
class SelfAttention(nn.Module):

    def __init__(self, dim, num_heads, dropout=0.):
        super(SelfAttention, self).__init__()
        self.wq = nn.Sequential(
             nn.Linear(dim, dim),
             nn.Dropout(p=dropout)
            )
        self.wk = nn.Sequential(
             nn.Linear(dim, dim),
             nn.Dropout(p=dropout)
            )
        self.wv = nn.Sequential(
             nn.Linear(dim, dim),
             nn.Dropout(p=dropout)
            )
        # MultiheadAttention
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)

    def forward(self, x):
        query = self.wq(x)
        key = self.wk(x)
        value = self.wv(x)
        att, _ = self.attn(query, key, value)
        out = att + x
        return out


class EMA(nn.Module):
    def __init__(self, channels, factor=1):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class CrossAttention(nn.Module):
    def __init__(self, input_dim_a, input_dim_b, hidden_dim):
        super(CrossAttention, self).__init__()
        self.linear_a = nn.Linear(input_dim_a, hidden_dim)
        self.linear_b = nn.Linear(input_dim_b, hidden_dim)
        self.pool = nn.AdaptiveAvgPool1d(128)

    def forward(self, input_a, input_b):
        input_a = self.pool(input_a.permute(0, 2, 1)).permute(0, 2, 1)
        input_b = self.pool(input_b.permute(0, 2, 1)).permute(0, 2, 1)
        mapped_a = self.linear_a(input_a)  # (batch_size, seq_len_a, hidden_dim)
        mapped_b = self.linear_b(input_b)  # (batch_size, seq_len_b, hidden_dim)
        scores = torch.matmul(mapped_a, mapped_b.transpose(1, 2))  
        attentions_a = torch.softmax(scores, dim=-1)  
        attentions_b = torch.softmax(scores.transpose(1, 2), dim=-1)  
        output_a = torch.matmul(attentions_b, input_b)  # (batch_size, seq_len_a, input_dim_b)
        output_b = torch.matmul(attentions_a.transpose(1, 2), input_a) 
        return output_a, output_b


class Fea_extractor(nn.Module):
    def __init__(self, embed_dim, layer=1, num_head=8, device='cuda'):
        super(Fea_extractor, self).__init__()
        self.layer = layer
        self.attn_drug = SelfAttention(dim=embed_dim, num_heads=num_head) 
        self.attn_protein = SelfAttention(dim=embed_dim, num_heads=num_head)
        self.ema = EMA(channels=1) 
        self.cross_attention = CrossAttention(input_dim_a=128,input_dim_b=128,hidden_dim=128)

    def forward(self, drug, protein):
        drug = self.attn_drug(drug) 
        protein = self.attn_protein(protein)
        drug = self.ema(drug.unsqueeze(1))  
        protein = self.ema(protein.unsqueeze(1))  
        v_d = reduce(drug, 'B H W C -> B W C', 'max') 
        v_p = reduce(protein, 'B H W C -> B W C', 'max') 
        v_d, v_p = self.cross_attention(v_d, v_p)
        v_d = reduce(v_d, 'B H W -> B H', 'max')
        v_p = reduce(v_p, 'B H W -> B H', 'max')
        f = torch.cat((v_d, v_p), dim=1) 
        return f, v_d, v_p, None

