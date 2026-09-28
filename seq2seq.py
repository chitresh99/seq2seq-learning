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


#####################################
# Decoding
#####################################


@torch.no_grad()
def greedy_decode(model, src, cfg):
    model.eval()
    src_lengths = torch.tensor([src.size(1)])
    state = model.encoder(src, src_lengths)
    token = torch.tensor([[SOS]])
    out = []

    for _ in range(cfg.max_decode_len):
        logits, state = model.decoder(token, state)
        token = logits[:, -1].argmax(dim=-1, keeptime=True)
        tid = token.item()
        if tid == EOS:
            break
        out.append(tid)

    return out


@torch.no_grad()  # no gradients needed at inference
def beam_search(model, src, cfg):
    model.eval()
    src_lengths = torch.tensor(
        [src.size(1)]
    )  # one sentence, so its length = number of columns
    state = model.encoder(src, src_lengths)  # (h, c): summary of the source

    beams = [([], 0.0, state)]  # each beam: (tokens so far, total log prob, lstm state)
    finished = []  # complete sentences (ended with <eos>)

    for _ in range(cfg.max_decode_len):
        candidates = []
        for tokens, score, st in beams:
            last = torch.tensor(
                [[tokens[-1] if tokens else SOS]]
            )  # feed last word, or <sos> at the very start
            logits, new_state = model.decoder(last, st)  # each beam uses ITS OWN state
            log_probs = torch.log_softmax(
                logits[0, -1], dim=-1
            )  # scores -> log probs (add instead of multiply)

            topk = torch.topk(
                log_probs, cfg.beam_width
            )  # best B next words for this beam
            for lp, idx in zip(topk.values.tolist(), topk.indices.tolist()):
                candidates.append(
                    (tokens + [idx], score + lp, new_state)
                )  # extend sentence, add its log prob

        candidates.sort(key=lambda x: x[1], reverse=True)  # best total score first

        beams = []
        for tokens, score, st in candidates:
            if tokens[-1] == EOS:
                finished.append((tokens[:-1], score))  # done: set aside, drop the <eos>
            else:
                beams.append((tokens, score, st))
            if len(beams) == cfg.beam_width:  # stop once we have B LIVE beams
                break

        if not beams:
            break

        if finished:
            best_finished = max(s for _, s in finished)
            if (
                best_finished >= beams[0][1]
            ):  # scores only go down, so no live beam can win now
                break

    finished += [
        (t, s) for t, s, _ in beams
    ]  # unfinished beams still count if we hit the length cap
    return max(finished, key=lambda x: x[1])[
        0
    ]  # tokens of the highest-scoring sentence


#####################################
# Evaluation
#####################################


def exact_match(model, pairs, cfg, src_vocab, tgt_vocab, decoder_fn):
    correct = 0
    for pair in pairs:
        src_ids, _ = tensorize(pair, src_vocab, tgt_vocab, cfg.reverse_source)
        src = torch.tensor([src_ids])
        predicted = tgt_vocab.decode(decoder_fn(model, src, cfg))
        if predicted == pair[1]:
            correct += 1
    return correct / len(pairs)


#####################################
# Checks
#####################################


def check_shapes(model, src_vocab, tgt_vocab, cfg):
    """Assert the handoff is wired correctly.

    Most bugs in this architecture are shape errors at the encoder-decoder
    boundary, or an off-by-one in the input/label offset. Both are cheap to
    catch and expensive to debug later.
    """
    batch, src_len, tgt_len = 4, 7, 9
    src = torch.randint(4, len(src_vocab), (batch, src_len))
    tgt_in = torch.randint(4, len(tgt_vocab), (batch, tgt_len))
    src_lengths = torch.full((batch,), src_len)  # every fake sentence has length 7

    h, c = model.encoder(src, src_lengths)
    assert h.shape == (cfg.num_layers, batch, cfg.hidden_dim), h.shape
    assert c.shape == (cfg.num_layers, batch, cfg.hidden_dim), c.shape

    logits = model(src, src_lengths, tgt_in)
    assert logits.shape == (batch, tgt_len, len(tgt_vocab)), logits.shape

    fixed_vector_size = cfg.num_layers * 2 * cfg.hidden_dim
    print(f"  encoder state   {tuple(h.shape)} x2 (h and c)")
    print(f"  logits          {tuple(logits.shape)}")
    print(
        f"  fixed vector    {cfg.num_layers} layers x 2 states x "
        f"{cfg.hidden_dim} units = {fixed_vector_size} numbers per sentence"
    )
    print(f"  parameters      {sum(p.numel() for p in model.parameters()):,}")


def check_overfit(cfg, src_vocab, tgt_vocab, train_pairs):
    """Can the model memorize 20 examples?

    If it cannot, something is structurally broken and no amount of training on
    the full set will help. This fails in seconds rather than minutes.
    """
    # 20 pairs in one batch means one optimizer step per epoch, so this needs
    # far more epochs than the full run to reach a comparable number of steps.
    small_cfg = Config(**{**cfg.__dict__, "epochs": 600, "batch_size": 20})
    model = Seq2Seq(len(src_vocab), len(tgt_vocab), small_cfg)
    subset = train_pairs[:20]
    train(model, subset, small_cfg, src_vocab, tgt_vocab, log_every=200)
    acc = exact_match(model, subset, small_cfg, src_vocab, tgt_vocab, greedy_decode)
    print(f"  memorized {acc:.0%} of 20 training pairs")
    return acc


#####################################
# Main
#####################################

def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    train_pairs, valid_pairs, src_vocab, tgt_vocab = build_dataset(cfg)

    print("Data")
    s, t = train_pairs[0]
    print(f"  source          {' '.join(s)}   (len {len(s)})")
    print(f"  target          {' '.join(t)}   (len {len(t)})")
    si, ti = tensorize(train_pairs[0], src_vocab, tgt_vocab, cfg.reverse_source)
    print(f"  source ids      {si}   reversed={cfg.reverse_source}, <eos> last")
    print(f"  target ids      {ti}   <sos> first, <eos> last")
    print(f"  vocab           {len(src_vocab)} source / {len(tgt_vocab)} target")
    print(f"  pairs           {len(train_pairs)} train / {len(valid_pairs)} held out")

    print("\nShapes")
    model = Seq2Seq(len(src_vocab), len(tgt_vocab), cfg)
    check_shapes(model, src_vocab, tgt_vocab, cfg)

    print("\nOverfit check")
    check_overfit(cfg, src_vocab, tgt_vocab, train_pairs)

    print("\nTraining")
    model = Seq2Seq(len(src_vocab), len(tgt_vocab), cfg)
    train(model, train_pairs, cfg, src_vocab, tgt_vocab)

    print("\nHeld-out exact match")
    greedy = exact_match(model, valid_pairs, cfg, src_vocab, tgt_vocab,
                         greedy_decode)
    beam = exact_match(model, valid_pairs, cfg, src_vocab, tgt_vocab,
                       beam_search)
    print(f"  greedy          {greedy:.1%}")
    print(f"  beam (B={cfg.beam_width})      {beam:.1%}")

    print("\nSamples")
    for pair in valid_pairs[:3]:
        src_ids, _ = tensorize(pair, src_vocab, tgt_vocab, cfg.reverse_source)
        out = greedy_decode(model, torch.tensor([src_ids]), cfg)
        print(f"  source          {' '.join(pair[0])}")
        print(f"  reference       {' '.join(pair[1])}")
        print(f"  predicted       {' '.join(tgt_vocab.decode(out))}")
        print()


if __name__ == "__main__":
    main()
