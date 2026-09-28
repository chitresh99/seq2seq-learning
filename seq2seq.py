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
    num_layers: int = 4
    hidden_dim: int = 128
    embed_dim: int = 128

    batch_size: int = 128
    lr_init: float = 1.0
    lr_decay_factor: float = 0.5
    clip_threshold: float = 5.0
    init_range: float = 0.08

    epochs: int = 40
    num_halvings: int = 5

    train_pairs: int = 6000
    valid_pairs: int = 300
    min_phrases: int = 2
    max_phrases: int = 5

    reverse_source: bool = True

    beam_width: int = 2
    max_decode_len: int = 24

    seed: int = 0


#####################################
# Data
#####################################

SRC_ADJ = ["red", "big", "small", "old", "new", "green", "cold", "bright"]
SRC_NOUN = ["cat", "house", "tree", "river", "bird", "stone", "road", "field"]

TGT_ADJ = {
    "red": "rouge",
    "big": "grand",
    "small": "petit",
    "old": "vieux",
    "new": "neuf",
    "green": "vert",
    "cold": "froid",
    "bright": "clair",
}

TGT_NOUN = {
    "cat": ["le", "chat"],
    "house": ["la", "maison"],
    "tree": ["arbre"],
    "river": ["la", "riviere"],
    "bird": ["oiseau"],
    "stone": ["la", "pierre"],
    "road": ["route"],
    "field": ["le", "champ"],
}

PAD, SOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<sos>", "<eos>", "<unk>"]


class Vocab:
    """Token <-> id mapping.
    The paper keeps the top 160k source and 80k target words
    by frequency and maps everything below that to <unk>.
    Here the synthetic vocabulary is small and closed, so nothing falls
    below the cut, but the <unk> id exists.
    """

    def __init__(self, tokens):
        self.itos = list(SPECIALS) + sorted(set(tokens) - set(SPECIALS))
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)  # returns the vocabulary size

    def encode(self, tokens):
        return [self.stoi.get(t, UNK) for t in tokens]

    def decode(self, ids):
        return [self.itos[i] for i in ids]


def make_pair(rng, min_phrases, max_phrases):
    """Generate one (source tokens, target tokens) pair."""
    n = rng.randint(min_phrases, max_phrases)
    src, tgt = [], []
    for _ in range(n):
        adj = rng.choice(SRC_ADJ)
        noun = rng.choice(SRC_NOUN)
        src += [adj, noun]
        tgt += TGT_NOUN[noun] + [TGT_ADJ[adj]]  # noun phrase first, adj after
    return src, tgt


def build_dataset(cfg):
    rng = random.Random(cfg.seed)
    train = [
        make_pair(rng, cfg.min_phrases, cfg.max_phrases)
        for _ in range(cfg.train_pairs)
    ]
    rng_valid = random.Random(cfg.seed + 10_000)
    valid = [
        make_pair(rng_valid, cfg.min_phrases, cfg.max_phrases)
        for _ in range(cfg.valid_pairs)
    ]
    src_vocab = Vocab(SRC_ADJ + SRC_NOUN)  # convert source sentence to id's
    tgt_vocab = Vocab(
        [t for v in TGT_NOUN.values() for t in v] + list(TGT_ADJ.values())
    )  # convert target sentence into id's for training and to turn model's output ids back into words
    return train, valid, src_vocab, tgt_vocab


def tensorize(pair, src_vocab, tgt_vocab, reverse_source):
    src_tokens, tgt_tokens = pair
    src_ids = src_vocab.encode(src_tokens)
    if reverse_source:
        src_ids = src_ids[::-1]
    src_ids = src_ids + [EOS]
    tgt_ids = [SOS] + tgt_vocab.encode(tgt_tokens) + [EOS]
    return src_ids, tgt_ids

def make_batches(pairs, cfg, src_vocab, tgt_vocab, shuffle=True):
    data = [tensorize(p, src_vocab, tgt_vocab, cfg.reverse_source)
            for p in pairs]
    data.sort(key=lambda x: len(x[0]))  # sorts all examples by source length, shortest first
    batches = []
    for i in range(0, len(data), cfg.batch_size):
        chunk = data[i:i + cfg.batch_size]
        src_len = max(len(s) for s, _ in chunk)
        tgt_len = max(len(t) for _, t in chunk)
        src = torch.full((len(chunk), src_len), PAD, dtype=torch.long)
        tgt = torch.full((len(chunk), tgt_len), PAD, dtype=torch.long)
        for j, (s, t) in enumerate(chunk):
            src[j, :len(s)] = torch.tensor(s)
            tgt[j, :len(t)] = torch.tensor(t)
        src_lengths = torch.tensor([len(s) for s, _ in chunk], dtype=torch.long)
        batches.append((src, tgt, src_lengths))
 
    if shuffle:
        if rng is None:
            rng = random.Random(cfg.seed)
        rng.shuffle(batches)
    return batches

