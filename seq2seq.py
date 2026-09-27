import torch
import random
from dataclasses import dataclass

import torch
import torch.nn as nn

#####################################
# Config
#####################################

@dataclass
class Config:
    num_layers : int = 4
    hidden_dim : int = 128
    embed_dim : int = 128

    batch_size : int = 128
    lr_init : float = 1.0
    lr_decay_factor: float = 0.5
    clip_threshold: float = 5.0
    init_range: float = 0.08

    epochs: int = 40
    num_halvings:int = 5

    train_pairs: int = 6000
    valid_pairs: int = 300
    min_phrases: int = 2
    max_phrases: int = 5

    reverse_source: bool = True

    beam_width: int = 2
    max_decode_len: int = 24

    seed:int = 0


    