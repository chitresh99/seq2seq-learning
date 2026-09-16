# Sequence to Sequence Learning with Neural Networks

Sutskever, Vinyals, Le (2014). NeurIPS. arXiv:1409.3215.

Reading notes and implementation reference.

---

## Contents

1. [First intuition](#1-first-intuition)
2. [Technical explanation](#2-technical-explanation)
3. [The model](#3-the-model)
4. [The math](#4-the-math)
5. [Data](#5-data)
6. [Training](#6-training)
7. [Inference](#7-inference)
8. [Results](#8-results)
9. [Conclusions the paper draws](#9-conclusions-the-paper-draws)
10. [Where the gap lies](#10-where-the-gap-lies)
11. [What came next](#11-what-came-next)
12. [Summary paragraph](#12-summary-paragraph)
13. [Questions and answers](#13-questions-and-answers)

---

## 1. First intuition

Suppose you want to translate English to French with a neural network. You immediately hit a problem that has nothing to do with language.

A neural network is a fixed function. It takes a vector of some size and returns a vector of some size, and both sizes are decided when you build it. That is fine for classifying an image, where one picture maps to one label. It is useless for translation, where the input is a sentence of unknown length and the output is a different sentence of a different unknown length.

The idea in this paper is to split the problem in two.

First, read the entire English sentence and squeeze everything you understood about it into a single fixed size vector. This is a compression step. It does not matter that the input was 6 words or 60, what comes out is the same size.

Second, take that vector and unroll it back into a French sentence, one word at a time, until you decide to stop. Because you decide when to stop, the output length is free.

Two networks, one vector passing between them. That is the entire architecture.

The compression is the bold part. You are asserting that everything needed to translate a sentence can survive being packed into a fixed block of numbers. The paper's contribution is showing that this works well enough to beat the mature statistical systems of the day. The rest of the field's next five years is largely about the fact that it does not always work, and what to do instead.

---

## 2. Technical explanation

The input is a token sequence `x1 ... xT`. The output is a token sequence `y1 ... yT'`. Note the primes: the two lengths are independent, and neither is known in advance.

**Encoder.** An LSTM reads the source one token at a time. At each step it combines the incoming token with the state it has carried so far and produces a new state. It emits no output. When it reads the end of sequence marker, its final state is taken as the representation of the whole input. Call it `v`.

**Decoder.** A second LSTM, with its own separate weights, is initialized with `v` as its starting state. It generates one token at a time. At each step it produces a probability distribution over the target vocabulary, a token is selected, and that token becomes the input for the next step. It halts when it emits the end of sequence marker.

The only channel between the two halves is `v`. The decoder never sees the source tokens. Everything it will ever know about the input has to survive that compression.

The practical consequence, which the paper leans on, is that the source sentence is reversed before encoding. The target is not. This costs nothing and is worth several BLEU points.

---

## 3. The model

| component | value |
|---|---|
| encoder | 4 layer LSTM |
| decoder | 4 layer LSTM, separate weights |
| hidden units per layer | 1000 |
| word embedding dimension | 1000 |
| source vocabulary | 160,000 |
| target vocabulary | 80,000 |
| total parameters | 384M, of which 64M are pure recurrent connections (32M encoder, 32M decoder) |
| output layer | naive softmax over 80,000 words at each step |

Depth mattered. Each additional layer reduced perplexity by nearly 10 percent, which the paper attributes to the larger hidden state.

### Why two separate LSTMs

They could have shared weights. Using two roughly doubles parameters at negligible compute cost, lets each half specialize, and makes it natural to train one model on several language pairs.

### What actually crosses between them

An LSTM layer carries two state vectors, the hidden state `h` and the cell state `c`, each 1000 wide. There are 4 layers. So the handoff is:

```
4 layers x 2 states x 1000 = 8000 real numbers
```

1000 is the width of one state vector. 8000 is the size of the complete fixed representation. The paper describes the sentence representation in exactly these terms.

The number that matters is that 8000 is constant. A six word sentence and a sixty word sentence both get 8000 floats.

### Shapes, for implementation

| stage | shape |
|---|---|
| source IDs | `(B, S)` |
| source embeddings | `(B, S, 1000)` |
| encoder final h, c | `(4, B, 1000)` each |
| decoder outputs | `(B, T, 1000)` |
| output logits | `(B, T, 80000)` |

Most bugs in this architecture are shape errors at the handoff, or an off by one in the decoder input and label offset. Assert both.

---

## 4. The math

### The plain RNN, which the paper shows and then abandons

```
h_t = sigm( W_hx  x_t  +  W_hh  h_(t-1) )
y_t = W_yh  h_t
```

Read the superscripts destination first: `W_hx` is the matrix going from `x` to `h`.

Line 1 updates memory. It transforms the current input, transforms the previous state, adds them, and squashes the result. The recurrence lives in `h_(t-1)` appearing on the right hand side, which is what makes every step depend on all the steps before it.

Line 2 produces output. It maps the hidden state to one score per vocabulary word. There is no nonlinearity here, so these are raw logits and softmax is applied afterward.

The encoder runs line 1 only. The decoder runs both.

The paper presents this and moves on for two reasons. The equations assume one output per input, aligned step by step, which translation violates. And `W_hh` gets applied repeatedly when gradients travel backward, so it either vanishes or explodes over long distances. LSTMs replace the plain update with gating machinery designed to fix exactly that.

### The decoder's objective

```
                          T'
p(y1 ... yT' | x1 ... xT) = ∏  p(y_t | v, y1 ... y_(t-1))
                          t=1
```

**Left side:** the goal. One number, the probability of the whole target sentence given the whole source sentence.

**Right side:** the method. A product of per word probabilities, each of which the model can actually compute as one row of its softmax output.

Written out for a concrete example:

```
p(Marie admire Jean | Mary admires John)
    = p(Marie  | v)
    x p(admire | v, Marie)
    x p(Jean   | v, Marie, admire)
    x p(<EOS>  | v, Marie, admire, Jean)
```

Two things this equation encodes. The source tokens appear on the left but vanish on the right, replaced entirely by `v`. That substitution is the bottleneck, stated formally. And the conditioning set grows by one word per step, which is why the decoder must generate left to right and feed itself its own output.

### Why logs

The left side is the quantity you care about, and it is unusably small. Thirty factors below 1 multiplied together lands near 10^-40, which underflows to zero in floating point.

Taking the log of both sides fixes it, and by the rule `log(a x b) = log(a) + log(b)` the product becomes a sum:

```
log p(y1 ... yT' | x) = Σ log p(y_t | v, y1 ... y_(t-1))
```

Checked numerically:

```
product form:  0.93 x 0.88 x 0.80 x 0.91 = 0.596
log form:     -0.07 + -0.13 + -0.22 + -0.09 = -0.51
log(0.596) = -0.51
```

Log is monotonic, so ranking by log probability and ranking by probability give the same order. That is what allows beam search and rescoring to run entirely in log space.

### Training objective

```
1/|S|  Σ  log p(T|S)
     (T,S) ∈ S
```

`S` is the training set of pairs, `T` the target, `S` the source. Go through every pair, score the log probability the model assigns to the correct translation, and average.

The average rather than the sum keeps the magnitude independent of dataset size. Negate it and you have cross entropy, which is what you minimize.

### Decoding objective

```
T_hat = arg max  p(T|S)
          T
```

`arg max` returns the argument, meaning the sentence itself, not the maximum value.

This is the hard one. With an 80,000 word vocabulary and length 20, the search space is 80000^20. You cannot enumerate it. Beam search is an approximate solver, with no guarantee of finding the true maximum.

---

## 5. Data

WMT'14 English to French. A 12M sentence pair subset, 348M French words and 304M English words. Evaluation on newstest2014, 3003 sentence pairs.

The raw format is two line aligned plain text files. Line n of one is the translation of line n of the other. No alignments, no parse trees, no phrase tables, no linguistic annotation of any kind. This is what "minimal assumptions on the sequence structure" means concretely.

### One pair through the pipeline

Raw:

```
en: We must act now.
fr: Nous devons agir maintenant.
```

Tokenize:

```
src = ["We", "must", "act", "now", "."]
tgt = ["Nous", "devons", "agir", "maintenant", "."]
```

Vocabulary lookup, out of vocabulary tokens become `UNK`:

```
src = [22, 66, 1893, 412, 7]
tgt = [31, 507, 904, 288, 7]
```

Reverse the source, add boundary tokens:

```
src = [7, 412, 1893, 66, 22, 1]        reversed, <EOS> appended
tgt = [2, 31, 507, 904, 288, 7, 1]     <SOS> prepended, <EOS> appended
```

Note the asymmetry. The source gets `<EOS>` only, since the encoder needs to know when to stop reading. The target gets both, since the decoder needs `<SOS>` as its first input and `<EOS>` as its halt signal.

Reverse before appending `<EOS>`, not after. You want `<EOS>` to be the last thing the encoder reads, because its state at that point becomes `v`.

### Where reversal happens

In preprocessing, not inside the model. The encoder is an ordinary left to right LSTM with no reversal logic. You flip the source token list when building the dataset and the encoder simply reads that as its input order.

```python
src_ids = src_ids[::-1] + [EOS]
tgt_ids = [SOS] + tgt_ids + [EOS]
```

Offline (write flipped IDs to disk once) and on the fly (flip in `__getitem__`) are equivalent in effect. On the fly behind a flag is preferable for reproduction, since you want to run the reversed against unreversed ablation without rebuilding your data files.

The thing that matters wherever you put it: the model must never see unreversed input at inference if it was trained on reversed. Tie both paths to one flag.

### On UNK

The criterion is frequency rank, not absence from the corpus. Top 160,000 source words and top 80,000 target words are kept, everything below becomes `UNK`.

Two reasons for the cut. The output softmax runs over the entire target vocabulary at every decoding step, so size is expensive. And a word appearing three times in 12M sentences receives three gradient updates and never learns a usable embedding.

The cut is by type, so a word below the rank threshold has every instance replaced and the model never sees it once. `UNK` is one symbol standing in for tens of thousands of distinct words, so a rare place name, a technical term and a typo are all indistinguishable.

Two failure modes. On input, an unseen word arrives as `UNK` and its identity is gone. On output, if the correct next token is a rare proper noun, the model cannot emit it at all, because it is not in the output vocabulary. This is what the abstract means by BLEU being penalized on out of vocabulary words: the 34.8 was achieved while handing away points the unlimited vocabulary SMT baseline did not lose the same way.

The vocabulary asymmetry reflects an asymmetry of cost. Reading an unknown word gives you one fuzzy input. Generating one gives you a guaranteed wrong output token. The output side is also where the softmax expense lives.

The field's eventual answer was to stop using words as the unit. Byte pair encoding (Sennrich et al., 2015) splits rare words into known subword fragments, which is why `UNK` does not appear in modern systems.

---

## 6. Training

### Hyperparameters

| setting | value |
|---|---|
| initialization | uniform [-0.08, 0.08] |
| optimizer | SGD, no momentum |
| learning rate | 0.7 fixed for 5 epochs, then halved every half epoch |
| total epochs | 7.5 |
| batch size | 128 sequences |
| gradient clipping | hard norm constraint at threshold 5 |
| hardware | 8 GPUs, one layer per GPU |
| wall clock | about 10 days |

The gradient is divided by the batch size 128 first, then `s = ||g||` is computed. If `s > 5`, the gradient is rescaled to `5g/s`. The paper notes LSTMs largely avoid vanishing gradients but can still explode, which is what this guards against.

The paper also sorts sentences of similar length into the same minibatch. Most sentences are 20 to 30 tokens but some exceed 100, so a batch of 128 randomly drawn sentences wastes most of its computation on padding. Length bucketing gives a 2x speedup.

### One training batch, end to end

1. Load the aligned pair, tokenize.
2. Map tokens to IDs, out of vocabulary becomes `UNK`.
3. Reverse the source, add boundary tokens.
4. Batch and pad, having sorted by similar length.
5. Embed the source. Each ID indexes a row of the embedding table, giving `(B, S, 1000)`.
6. Encoder forward. Per timestep outputs are discarded, the final `(h, c)` per layer is kept.
7. Hand off. Copy the encoder's final `(h, c)` in as the decoder's initial `(h, c)`.
8. Build decoder input and label by offset.
9. Decoder forward, all positions in one call.
10. Project each output to vocabulary scores with the linear layer.
11. Cross entropy against the labels, pad positions masked out.
12. Backward, clip, step.

### Teacher forcing

The target tensor serves two roles, offset by one position:

```python
decoder_input  = tgt_ids[:-1]    # [<SOS>, Nous, devons, agir, maintenant, .]
decoder_target = tgt_ids[1:]     # [Nous, devons, agir, maintenant, ., <EOS>]
```

At each position the input is the true previous token and the label is the token to predict. That slice by one is the entire implementation of teacher forcing.

Laid out as a table for the example pair:

| position | model sees | model must predict |
|---|---|---|
| 1 | `<SOS>` | Marie |
| 2 | Marie | admire |
| 3 | admire | Jean |
| 4 | Jean | `<EOS>` |

The consequence: if the model gets position 1 wrong, position 2 still receives the correct word. Training never has to recover from its own errors.

Because the whole true target is available, the decoder runs as one call over all timesteps rather than a Python loop.

### The embedding table

The lookup is deterministic array indexing. ID 6 always fetches row 6.

What is random is the table's initial contents. Every row starts as small random numbers, so embeddings are initially meaningless. The rows are trainable parameters, gradients flow into them, and after training each row holds a learned representation.

### The output projection

The decoder gives 1000 numbers per position. You need a score for each of 80,000 target words. So every target word has its own list of 1000 weights, and scoring a word means multiplying position by position against the decoder state and summing.

Small worked example with 4 dimensions and a 7 word vocabulary:

```
decoder state:    [0.33, -0.12, 0.44, 0.10]
Marie's weights:  [2.0,  -1.0,  5.0,  3.0]

0.33 x  2.0 =  0.66
-0.12 x -1.0 =  0.12
0.44 x  5.0 =  2.20
0.10 x  3.0 =  0.30
              -----
               3.28     Marie's score
```

Repeat for all vocabulary entries, then softmax to get probabilities. Stacking those per word weight lists into one table and doing all of them at once is a matrix multiply, which is all a linear layer is.

At the real scale this table is 1000 x 80,000, which is 80M parameters in one layer, evaluated at every decoding step. That expense is exactly why the target vocabulary was capped at 80k while the source could afford 160k.

### In log space

You never materialize raw probabilities. Apply `log_softmax` rather than `softmax`, gather the entries for the target words, and sum. Negate because optimizers minimize.

```python
log_probs = F.log_softmax(logits, dim=-1)
loss = -log_probs.gather(-1, targets).sum()
```

Do not compute softmax and then take the log yourself, that reintroduces the underflow you were avoiding.

---

## 7. Inference

Steps 1 through 7 above are identical. Step 9 changes completely, because no target exists to feed.

```
state = encoder final (h, c)
token = <SOS>
loop:
    out, state = decoder_step(embed(token), state)
    logits = projection(out)
    token = pick from logits
    emit token
until token == <EOS> or max length reached
```

Two differences from training. This is a genuine sequential loop, because each input depends on the previous output. And `pick from logits` is where beam search lives.

### The exposure problem

| | training | inference |
|---|---|---|
| decoder input at step 2 | true word from the label | whatever step 1 emitted |
| execution | one parallel call | sequential loop |
| stopping | target length is known | when `<EOS>` appears |

If the model emits a wrong token at step 1, training still feeds it the correct word at step 2. Inference feeds it its own error, and the output derails from there. The model is being asked at test time to condition on sequences it never saw during training.

### Beam search

Instead of committing to the argmax at each step, keep the B highest scoring partial sequences alive, each with its own LSTM state and running cumulative log probability.

"Partial" means generated up to the current step. At step 2 every candidate is two words long, at step 3 three words. They grow together, one token per step.

At each step, every one of the B candidates is extended by every vocabulary word, giving B x V continuations. Score them all, sort, keep the top B, discard the rest permanently. The beam width never grows.

```
step 1, expand <SOS>:
  log p(Jean)  = -0.6
  log p(Marie) = -0.9      kept, though not the argmax

step 2, expand both, keep top 2:
  Jean admire   : -0.6 + -0.5 = -1.1
  Marie admire  : -0.9 + -0.2 = -1.1

step 3:
  Jean admire Marie : -1.4
  Marie admire Jean : -1.2      wins
```

The path that lost at step 1 wins overall, because later evidence revealed it was better. Greedy decoding could never find this, because `Marie` was already discarded.

A hypothesis leaves the beam when it emits `<EOS>` and is set aside as a complete candidate.

Why it is approximate: the true argmax would require keeping every path alive to the end, since a path that looks bad at step 3 might be best at step 12. Pruning to B throws away paths you can never recover. Greedy decoding is the extreme case, B = 1.

The paper reports that beam size 2 captures most of the available gain.

---

## 8. Results

### Table 1, LSTM alone on newstest2014

| method | BLEU |
|---|---|
| Bahdanau et al. | 28.45 |
| Baseline SMT system | 33.30 |
| Single forward LSTM, beam 12 | 26.17 |
| Single reversed LSTM, beam 12 | 30.59 |
| Ensemble of 5 reversed LSTMs, beam 1 | 33.00 |
| Ensemble of 2 reversed LSTMs, beam 12 | 33.27 |
| Ensemble of 5 reversed LSTMs, beam 2 | 34.50 |
| Ensemble of 5 reversed LSTMs, beam 12 | **34.81** |

**Reversal is the single biggest lever.** 26.17 to 30.59, same architecture and same beam, is 4.4 BLEU for one line in the data pipeline. Larger than the gain from ensembling or from any beam width change.

**Ensembling matters more than beam width.** Five models at beam 1, meaning no search at all, score 33.00, beating one model at beam 12. Once you have the ensemble, going from beam 2 to beam 12 buys 0.31 BLEU. The paper's caption makes the deployment point explicitly: 5 LSTMs at beam 2 is cheaper than 1 LSTM at beam 12 and scores better.

**The headline 34.8 is an ensemble result.** The abstract states it without qualification. A single model gets 30.59, which is below the 33.30 baseline. The honest statement is that a single seq2seq model did not beat phrase based SMT, the ensemble did.

### Table 2, LSTM combined with SMT

| method | BLEU |
|---|---|
| Baseline SMT system | 33.30 |
| Cho et al. | 34.54 |
| Best WMT'14 result | 37.0 |
| Rescoring baseline 1000-best, single forward LSTM | 35.61 |
| Rescoring baseline 1000-best, single reversed LSTM | 35.85 |
| Rescoring baseline 1000-best, ensemble of 5 reversed | **36.5** |
| Oracle rescoring of the baseline 1000-best | about 45 |

Even a single LSTM, which cannot beat SMT on its own, adds 2.3 BLEU as a reranker. The two systems' errors are genuinely complementary.

The oracle number is the most informative one. Oracle means cheating: for each source sentence, pick whichever of the 1000 candidates actually scores best against the reference. That gives about 45. So the SMT system's n-best lists already contain translations worth 45 BLEU, and the best reranker of the day extracted only 36.5 of it. The gap between those is the honest measure of how much room was left in reranking, and it suggests reranking was a dead end relative to building better models directly.

### Reversal, measured separately

On the development setup the paper reports perplexity dropping from 5.8 to 4.7 and BLEU rising from 25.9 to 30.6 when the source is reversed.

### Why reversal helps

Count the steps between reading a source word and emitting its translation, on the combined encoder plus decoder timeline:

| pair | unreversed | reversed |
|---|---|---|
| Mary to Marie | 3 | 1 |
| admires to admire | 3 | 3 |
| John to Jean | 3 | 5 |
| total | 9 | 9 |

The average distance is unchanged. What changes is the distribution: "all medium" becomes "some easy, some hard."

Gradients attenuate with distance, so in the unreversed case every correspondence the model must learn carries weak signal. Reversal creates at least one near trivial connection at the sentence start, which is learned fast. Because the decoder conditions on its own previous outputs, getting the first word right is disproportionately valuable: a correct first token is a good input for predicting the second, and so on. Reversal buys a foothold early in training.

The paper presents this as a hypothesis and states it does not have a complete explanation.

### Terms used in reading these tables

A **baseline** is the existing answer to the problem, the reference point your method must beat for the result to mean anything. Same test set, same metric, and ideally a strong well tuned version of the prior method. Here the baseline is a real published system, not one built to lose, which is part of why the result was taken seriously.

An **ablation** removes a piece of your own model to see what it contributed. The forward versus reversed comparison is one.

An **oracle** is an upper bound computed by cheating. Not a baseline, it is the ceiling rather than the floor.

---

## 9. Conclusions the paper draws

**The result, framed at its sharpest.** A model with a limited vocabulary, making almost no assumptions about problem structure, beat an SMT system with an unlimited vocabulary. They are pointing out they won while handicapped on two fronts.

**Generalization.** If this worked for translation it should work for other sequence problems given enough data. This is the claim the paper is remembered for, it turned out correct, and it is why the architecture name outlived the translation result.

**A transferable principle from the reversal finding.** They state they were surprised by the size of the effect, then generalize: find a problem encoding with the greatest number of short term dependencies, because those make the learning problem simpler. That is a claim about how to set up problems for gradient based learning, not a claim about translation.

They also admit they could not train a plain RNN on the non reversed problem at all, and believe a plain RNN would train fine on the reversed version, but explicitly flag that they did not test it.

**Long sentences.** They state they were initially convinced the LSTM would fail on long sentences due to limited memory, and note that other researchers reported exactly that failure with similar models. Their reversed models did not show it. This is reported as an empirical observation with no mechanism offered, contradicting what others found.

**The invitation.** They stress the approach was relatively unoptimized, so further work should push higher. A deliberate signal that this is a starting point. The field took it.

Worth noticing about the writing: "we were surprised" appears twice, and there is an explicitly unverified belief. The conclusion is candid about what the authors do not understand, which is a good model for writing up a reproduction.

---

## 10. Where the gap lies

The vulnerable assumption is that an entire sentence of arbitrary length and complexity survives compression into one fixed size vector.

The abstract claims the LSTM did not have difficulty on long sentences, but that is a claim about the length distribution of one test set, and the paper offers no mechanism for why the capacity argument should not bite. Later work confirmed that fixed vector encoders degrade past roughly thirty words. The authors' own conclusion admits both the surprise and the contradiction with other groups' findings.

The reversal trick can be read as a symptom. If flipping the input order is worth 4.4 BLEU, the optimization is fragile in a way the architecture is not accounting for. The same information crosses the same bottleneck either way, only the gradient path lengths changed.

On paragraphs: the architecture is length agnostic, since nothing in it knows what a sentence is, but the representation is not. A paragraph gets the same 8000 floats as a sentence. State also resets between pairs, so the model has no notion of document context by construction. Document level translation with a fixed vector is possible in principle and poor in practice.

---

## 11. What came next

Bahdanau, Cho and Bengio (2015) attacked this assumption directly with attention: keep every encoder state instead of only the last, and let the decoder compute a weighted read over all of them at each step.

| | this paper | with attention |
|---|---|---|
| encoder states kept | only the last | all of them |
| size of what the decoder reads | fixed, 8000 floats | grows with source length |
| decoder access to source word 1 | through the bottleneck only | direct |

When the decoder is about to emit a word, it can place most of its weight on the encoder state for the corresponding source word and read from it directly. No bottleneck to survive, and no need for the reversal trick. That line runs to the Transformer.

**One line summary of the whole paper.** A fixed size vector is both the whole idea and the whole limitation. Everything else in the surrounding literature, reversal and beam search and reranking and attention, is either working around that constraint or removing it.

---

## 12. Summary paragraph

Before this paper, neural networks required fixed size inputs and outputs, which ruled out translation, since you cannot know the output length in advance. Sutskever, Vinyals and Le showed you could handle it with two LSTMs: one encodes the source sentence into a fixed dimensional vector, a second decodes the target from that vector, generating one token at a time until it emits an end of sequence marker. No alignment models, no phrase tables, no hand engineered pipeline, just one model trained end to end with SGD on raw token sequences. It worked at scale: 34.8 BLEU on WMT'14 English to French against a mature phrase based SMT system at 33.3, which was the first time a purely neural system beat statistical machine translation on a large scale task, and it did so while handicapped by a limited vocabulary the baseline did not have. Two findings stand out. Reversing the source sentence, which costs nothing, was worth around 4 BLEU, because it creates short range dependencies early in the sequence that make the optimization easier. And using the LSTM to rescore the SMT system's 1000-best list pushed the score to 36.5, showing the two approaches fail in complementary ways. The limitation is structural and the paper is candid about it: the entire source has to pass through one fixed vector of 8000 numbers regardless of sentence length, so the decoder never sees the source tokens at all. The authors expected this to break on long sentences and were surprised when it did not, while noting other groups had reported exactly that failure. Bahdanau et al. removed the constraint a year later with attention, and that line runs directly to the Transformer.

---

## 13. Questions and answers

**What problem does this paper solve?**

Neural networks required fixed size inputs and outputs. Translation needs variable length on both sides, with lengths that differ and are not known in advance. The paper shows an encoder LSTM plus a decoder LSTM handles this end to end, with no task specific structure.

**Why two LSTMs rather than one?**

Two roughly doubles the parameters at negligible compute cost, lets each half specialize on reading versus generating, and makes it natural to train on multiple language pairs. There is no deep reason it could not share weights, it simply works better not to.

**What exactly is the fixed vector?**

The encoder's final hidden and cell states across all layers. 4 layers times 2 states times 1000 units is 8000 real numbers. 1000 is the width of a single state vector, 8000 is the full handoff. The key property is that it does not grow with input length.

**How does the model know when to stop generating?**

It emits a special end of sequence token, which is in the target vocabulary like any other. The decoder loop halts when that token is produced. This is what decouples output length from input length.

**What is teacher forcing and why does it matter?**

During training, the decoder is fed the true previous target token at each step rather than its own prediction. It matters because inference cannot do this, so the model is trained on a distribution of inputs it never faces at test time. One wrong token early at inference corrupts everything downstream. That mismatch is the exposure problem, and beam search only mitigates it.

**What does beam search actually do?**

At each decoding step it keeps the B highest scoring partial sequences alive, scored by cumulative log probability, rather than committing to the single best token. Each candidate is extended by every vocabulary word, the resulting B times V continuations are scored, and only the top B survive. It lets a hypothesis that lost at step 1 win overall once later evidence arrives.

**Why log probabilities rather than probabilities?**

The probability of a full sentence is a product of per token probabilities, so for a thirty word sentence it lands near 10^-40 and underflows to zero. Logs turn the product into a sum that stays in a usable range. Log is monotonic so ranking is unaffected, which is what lets beam search and rescoring work entirely in log space.

**Why reverse the source sentence?**

It does not change the average distance between a source word and its translation, it changes the distribution. Reversal creates very short dependencies at the start of the sentence, which are learned quickly and give the model an early foothold. Since the decoder conditions on its own outputs, getting the beginning right is disproportionately valuable. Worth about 4 BLEU. The paper offers this as a hypothesis, not a proven mechanism, and I would treat it as an open question.

**Where does the reversal happen in the implementation?**

In preprocessing, not in the model. The encoder is an ordinary left to right LSTM. You flip the source token list when building the dataset, before appending the end of sequence token. Train and inference must use the same setting.

**What is rescoring and why did it beat both systems?**

The SMT system emits its 1000 best candidates. The LSTM scores each by log probability given the source, that score is combined with the SMT's own, and the list is reordered. The LSTM only judges here, it never generates. It beats both alone because the two systems make different kinds of errors, so the LSTM rescues good translations the SMT ranked too low while staying confined to candidates the SMT considered fluent.

**Did a single model beat the SMT baseline?**

No. A single reversed LSTM scored 30.59 against the baseline's 33.30. The 34.81 headline number is an ensemble of five. This is worth stating precisely, since the abstract does not qualify it.

**What is the oracle number in Table 2 and why does it matter?**

About 45 BLEU, obtained by cheating: for each sentence, picking whichever of the 1000 candidates scores best against the reference. It means the n-best lists already contained translations far better than any reranker extracted. The gap between 36.5 and 45 measures how much was left on the table, and suggests reranking was a weaker direction than building better models directly.

**What is the main weakness of this architecture?**

Everything about the source has to pass through one fixed size vector. The decoder never sees a source token. Capacity is constant regardless of whether the input is six words or sixty, so degradation on long inputs is the predicted outcome. The paper claims it did not observe this, but that is a claim about one test set's length distribution, and other groups reported the opposite with similar models.

**How did the field fix it?**

Attention. Keep all encoder states rather than only the last, and let the decoder compute a weighted read over them at each step, so what it reads scales with source length and it can access any source position directly. Bahdanau et al. 2015, and that line leads to the Transformer.

**What would you have tested that the paper did not?**

A sentence length ablation extending past their test distribution, to see where the fixed vector actually breaks. And whether reversal still helps once optimization is improved by other means, such as better initialization or a bidirectional encoder, which would separate a genuine architectural fix from a patch on a training artifact. The paper itself flags an untested belief about plain RNNs on reversed input, which is another cheap experiment.

**What is UNK and what does it cost?**

Any token outside the top 160,000 source or 80,000 target words by frequency becomes a single `UNK` symbol. On input it means the word's identity is lost. On output it means the model literally cannot produce that word, so it is a guaranteed mismatch against the reference. The abstract flags that BLEU was penalized for this. Subword tokenization, specifically byte pair encoding, removed the problem entirely a year later.

**Why is the source vocabulary larger than the target?**

Cost asymmetry. Reading an unknown word costs you one fuzzy input. Generating one costs you a guaranteed wrong output token. And the output side carries a 1000 by 80,000 projection evaluated at every decoding step, so enlarging it is expensive in a way the input embedding table is not.
