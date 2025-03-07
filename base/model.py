import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class TransformerConfig:
    vocab_size: int = 30000
    max_position_embeddings: int = 512
    d_model: int = 768
    n_layer: int = 12
    n_head: int = 12
    d_ff: int = 3072
    dropout: float = 0.1
    attn_dropout: float = 0.1
    layer_norm_epsilon: float = 1e-12
    pad_token_id: int = 0


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.n_head = config.n_head
        
        self.d_head = config.d_model // config.n_head
        
        assert self.d_model % self.n_head == 0, "d_model must be divisible by n_head"
        
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.scale = 1.0 / math.sqrt(self.d_head)
    
    def _split_heads(self, x):
        new_shape = x.size()[:-1] + (self.n_head, self.d_head)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)
    
    def _merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_shape = x.size()[:-2] + (self.d_model,)
        return x.view(*new_shape)
    
    def forward(self, x, attention_mask=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)
        
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        attn_weights = self.attn_dropout(attn_weights)
        
        context = torch.matmul(attn_weights, v)
        
        context = self._merge_heads(context)
        
        attn_output = self.out_proj(context)
        attn_output = self.resid_dropout(attn_output)
        
        return attn_output


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc_in = nn.Linear(config.d_model, config.d_ff)
        self.fc_out = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = F.gelu
    
    def forward(self, x):
        return self.dropout(self.fc_out(self.activation(self.fc_in(x))))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.ln_2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        
        self.attn = MultiHeadAttention(config)
        self.mlp = MLP(config)
    
    def forward(self, x, attention_mask=None):
        attn_output = x + self.attn(self.ln_1(x), attention_mask)
        
        output = attn_output + self.mlp(self.ln_2(attn_output))
        
        return output


class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.d_model)
        
        self.embd_dropout = nn.Dropout(config.dropout)
        
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        token_embeddings = self.wte(input_ids)
        position_embeddings = self.wpe(position_ids)
        
        hidden_states = token_embeddings + position_embeddings
        
        hidden_states = self.embd_dropout(hidden_states)
        
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, 1, 1, seq_len)
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask)
        
        hidden_states = self.ln_f(hidden_states)
        
        return hidden_states


class GPTLMHeadModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = GPTModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        self.lm_head.weight = self.transformer.wte.weight
    
    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.transformer(input_ids, attention_mask)
        logits = self.lm_head(hidden_states)
        return logits


transformerConfig = TransformerConfig(
    vocab_size=10000,
    max_position_embeddings=512,
    d_model=768,
    n_layer=6,
    n_head=12,
    d_ff=3072
)
    
GPT = GPTLMHeadModel(transformerConfig)
    
# batch_size = 2
# seq_len = 10
# input_ids = torch.randint(0, transformerConfig.vocab_size, (batch_size, seq_len))
# attention_mask = torch.ones(batch_size, seq_len)
    
# logits = model(input_ids, attention_mask)
# print(f"Logits shape: {logits.shape}")