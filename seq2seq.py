import torch
import random
import math
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
        make_pair(rng, cfg.min_phrases, cfg.max_phrases) for _ in range(cfg.train_pairs)
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
    data = [tensorize(p, src_vocab, tgt_vocab, cfg.reverse_source) for p in pairs]
    data.sort(
        key=lambda x: len(x[0])
    )  # sorts all examples by source length, shortest first
    batches = []
    for i in range(0, len(data), cfg.batch_size):
        chunk = data[i : i + cfg.batch_size]
        src_len = max(len(s) for s, _ in chunk)
        tgt_len = max(len(t) for _, t in chunk)
        src = torch.full((len(chunk), src_len), PAD, dtype=torch.long)
        tgt = torch.full((len(chunk), tgt_len), PAD, dtype=torch.long)
        for j, (s, t) in enumerate(chunk):
            src[j, : len(s)] = torch.tensor(s)
            tgt[j, : len(t)] = torch.tensor(t)
        src_lengths = torch.tensor([len(s) for s, _ in chunk], dtype=torch.long)
        batches.append((src, tgt, src_lengths))

    if shuffle:
        if rng is None:
            rng = random.Random(cfg.seed)
        rng.shuffle(batches)
    return batches


#####################################
# Model
#####################################


class Encoder(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, cfg.embed_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(
            cfg.embed_dim, cfg.hidden_dim, cfg.num_layers, batch_first=True
        )

    def forward(self, src, src_lengths):
        emb = self.embedding(src)  # (batch, src_len, embed)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        # outputs is deliberately unused; keeping it would be the first step
        # toward attention.
        _, (h, c) = self.lstm(packed)
        return h, c  # each (layers, batch, hidden)


class Decoder(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, cfg.embed_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(
            cfg.embed_dim, cfg.hidden_dim, cfg.num_layers, batch_first=True
        )
        # maps each hidden state to one score (logit) per target word.
        self.projection = nn.Linear(cfg.hidden_dim, vocab_size)

    def forward(self, tgt_in, state):
        emb = self.embedding(tgt_in)  # (batch, tgt_len, embed)
        outputs, state = self.lstm(emb, state)  # (batch, tgt_len, hidden)
        logits = self.projection(outputs)  # (batch, tgt_len, vocab)
        return logits, state


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(src_vocab_size, cfg)
        self.decoder = Decoder(tgt_vocab_size, cfg)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            nn.init.uniform_(p, -self.cfg.init_range, self.cfg.init_range)
        with torch.no_grad():
            self.encoder.embedding.weight[PAD].zero_()
            self.decoder.embedding.weight[PAD].zero_()

    def forward(self, src, src_lengths, tgt_in):
        state = self.encoder(src, src_lengths)
        logits, _ = self.decoder(tgt_in, state)
        return logits


#####################################
# Training
#####################################


def compute_loss(model, src, tgt):
    tgt_in = tgt[:, :-1]
    tgt_out = tgt[:, 1:]

    logits = model(src, tgt_in)
    loss_sum = nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        tgt_out.reshape(-1),
        ignore_index=PAD,
        reduction="sum",
    )
    num_tokens = (tgt_out != PAD).sum().item()
    return loss_sum / src.size(0), loss_sum.items(), num_tokens


def clip_gradients(model, threshold):
    return torch.nn.utils.clip_grad_norm_(model.parameters(), threshold).item()


def train(model, train_pairs, cfg, src_vocab, tgt_vocab, log_every=5):
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr_init)
    lr = cfg.lr_init

    decay_start = cfg.epochs - cfg.num_halvings

    for epoch in range(1, cfg.epochs + 1):
        if epoch > decay_start:
            lr *= cfg.lr_decay_factor
            for g in optimizer.param_groups:
                g["lr"] = lr

        model.train()
        batches = make_batches(train_pairs, cfg, src_vocab, tgt_vocab)
        total_loss, total_tokens = 0.0, 0

        for src, tgt in batches:  # one batch at a time
            optimizer.zero_grad()  # clear gradients from the previous batch (PyTorch accumulates them)
            loss, loss_sum, n_tokens = compute_loss(
                model, src, tgt
            )  # forward pass with teacher forcing; loss is scaled by batch size
            loss.backward()  # compute gradients of the loss w.r.t. every weight
            clip_gradients(
                model, cfg.clip_threshold
            )  # cap the gradient norm at 5 to stop exploding updates
            optimizer.step()  # update weights: w = w - lr * grad
            total_loss += loss_sum  # running total of token loss, for logging
            total_tokens += (
                n_tokens  # running count of real (non-pad) tokens, for per-token loss
            )

        if epoch % log_every == 0 or epoch == 1:
            per_token = total_loss / total_tokens
            print(
                f"  epoch {epoch:3d}  loss/token {per_token:6.4f}  "
                f"ppl {math.exp(per_token):8.3f}  lr {lr:.4f}"
            )

    return model
