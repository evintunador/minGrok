# Importing pytorch
import torch
import torch.nn as nn
from torch.nn import functional as F

import dataclasses 
from typing import Optional
from tokenizer import *

### BASE 
@dataclasses.dataclass
class Config:
    # v was defined earlier when we loaded TinyShakespeare. In Grok it's 131,072
    vocab_size: int = tokenizer.vocab_len

    # The maximum sequence length that this model might ever be used with.
    max_position_embeddings: int = 256 # in Grok it's 8,192

    # The number of layers in the model.
    num_layers: int = 4 # In Grok it's 64

    # The number of attention heads used in the attention layers of the model.
    num_attention_heads: int = 4 # In Grok it's 48

    # The number of key-value heads for implementing attention.
    num_key_value_heads: int = 1 # In Grok it's 8

    # The hidden size of the model, AKA the embedding dimension. Each token embedding vector will be this long
    hidden_size: int = 96 # In Grok it's 6,144

    # How much wider should the inner dimension of the experts be than the model's embedding dimension?
    embedding_multiplier_scale: int = 2 # In Grok it's roughly 5.33

    # how many experts?
    tot_num_experts: int = 4 # in Grok it's 8

    # how many active experts per token?
    chosen_num_experts: int = 2 # in Grok it's also 2

    # what amount of noise should be injected into the router during training?
    noise_std = 0.1 # the value for Grok has not been shared

    # When we create a loss to encourage all experts to be used, how should that loss be weighted?
    lambadada = 10 # Grok's value has not been shared
    # excuse my silly naming

    # The number of head dimensions
    head_dim: int = 24 # In Grok it's 128

    # The epsilon used by the rms normalization layers.
    rms_norm_eps: float = 1e-5 # this is to promote numerical stability & prevent dividing by 0

    # the scaling factor that determines the frequencies for the rotary positional encodings
    rope_theta = 100.0 # Grok and most models use 10,000
    # smaller models should use a smaller theta, but I'm just guessing here. 1000 might work too

    # whether to use a linear layer after normalization
    use_scale: bool = True # same in Grok

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # the dropout rate to use during training
    dropout = 0.05