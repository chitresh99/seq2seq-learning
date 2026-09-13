# Workflow

## Training, one batch

**1. Load and tokenize.** Pull the aligned pair, split into tokens.

**2. Map to IDs.** Vocabulary lookup, anything below the frequency cut becomes `UNK`.

**3. Reverse the source, add boundary tokens.**
```
src: [.,  now, act, must, We, <EOS>]
tgt: [<SOS>, Nous, devons, agir, maintenant, ., <EOS>]
```

**4. Batch and pad.** Sort by similar length first so padding is minimal.

**5. Embed the source.** Each ID indexes a row of the embedding table, so the batch becomes a tensor of shape `(batch, src_len, 1000)`.

**6. Encoder forward.** The 4-layer LSTM consumes the sequence left to right. It returns per-timestep outputs, which you throw away, and the final `(h, c)` for each layer, which is what you keep. That's your 8000 floats per sentence.

**7. Hand off.** Copy the encoder's final `(h, c)` in as the decoder's initial `(h, c)`. This is the only connection between the two halves.

**8. Build the decoder input and label by offset.**
```
input:  [<SOS>, Nous, devons, agir, maintenant, .]
label:  [Nous, devons, agir, maintenant, ., <EOS>]
```

**9. Decoder forward.** Embed the input, run the 4-layer LSTM starting from the handed-off state. Because you have the entire true target already, this runs as one call over all timesteps rather than a Python loop. Teacher forcing is free here, it's just a consequence of feeding the true sequence.

**10. Project to vocabulary.** A linear layer takes each 1000-dim decoder output to a vector of size 80,000. Shape becomes `(batch, tgt_len, 80000)`.

**11. Loss.** Cross-entropy against the labels, with pad positions masked out.

**12. Backward, clip, step.** Gradients flow back through the decoder, through the handoff, and into the encoder, so the encoder is trained purely by the signal arriving through that state. The paper clips gradient norms, which matters, LSTMs at this depth blow up without it.

## Inference, which is structurally different

Steps 1 to 7 are identical. Step 9 changes completely, because there's no target to feed.

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

Two differences worth being clear about. This is a genuine Python loop, one step at a time, because each input depends on the previous output. And `pick from logits` is where beam search lives: instead of one `state` and one `token`, you carry B of them, expand each, and keep the B best by cumulative log probability.

## The shapes, in one place

| stage | shape |
|---|---|
| source IDs | `(B, S)` |
| source embeddings | `(B, S, 1000)` |
| encoder final h, c | `(4, B, 1000)` each |
| decoder outputs | `(B, T, 1000)` |
| logits | `(B, T, 80000)` |

Most implementation bugs in this architecture are shape bugs at the handoff, or an off-by-one in the input/label offset. Worth asserting both explicitly the first time through.

## Simplified

Let me shrink everything so the numbers are readable. Toy setup: 1 layer instead of 4, hidden size 4 instead of 1000, vocabulary of 10 words instead of 80,000. Everything else is identical to the real thing.

## The vocabularies

```
English            French
0: <PAD>           0: <PAD>
1: <EOS>           1: <EOS>
2: <SOS>           2: <SOS>
3: <UNK>           3: <UNK>
4: Mary            4: Marie
5: admires         5: admire
6: John            6: Jean
```

## Step 1: one raw pair

```
Mary admires John   ->   Marie admire Jean
```

## Step 2: tokenize and look up IDs

```
src = [4, 5, 6]
tgt = [4, 5, 6]
```

(Coincidence that they match, the vocabularies are separate tables.)

## Step 3: reverse source, add boundary tokens

```
src = [6, 5, 4, 1]          John admires Mary <EOS>
tgt = [2, 4, 5, 6, 1]       <SOS> Marie admire Jean <EOS>
```

## Step 4: split the target into input and label

```
dec_in  = [2, 4, 5, 6]      <SOS> Marie admire Jean
dec_out = [4, 5, 6, 1]      Marie admire Jean <EOS>
```

Line them up vertically and the whole training objective becomes visible:

| position | model sees | model must predict |
|---|---|---|
| 1 | `<SOS>` | Marie |
| 2 | Marie | admire |
| 3 | admire | Jean |
| 4 | Jean | `<EOS>` |

## Step 5: embed the source

Each ID picks a row from a 4-wide table:

```
6 -> [ 0.21, -0.44,  0.90,  0.13]
5 -> [-0.32,  0.71,  0.05, -0.88]
4 -> [ 0.67,  0.02, -0.51,  0.40]
1 -> [ 0.00,  0.15,  0.22, -0.09]
```

## Step 6: encoder runs

```
h0 = [0, 0, 0, 0]

t=1  read John      h1 = [ 0.12, -0.30,  0.55,  0.08]
t=2  read admires   h2 = [-0.05,  0.41,  0.33, -0.60]
t=3  read Mary      h3 = [ 0.48,  0.19, -0.22,  0.31]
t=4  read <EOS>     h4 = [ 0.51,  0.23, -0.18,  0.29]   <- this is v
```

h1, h2, h3 are gone. Only h4 continues. (Plus the cell state c4, same shape, which I'm omitting for clarity but travels alongside.)

## Step 7: handoff

```
decoder s0 = h4 = [0.51, 0.23, -0.18, 0.29]
```

## Step 8: decoder runs, all positions at once

Because you already have the true `dec_in`, you don't loop. All four inputs go through together:

```
pos 1: embed(<SOS>)  -> s1 -> [ 0.33, -0.12,  0.44,  0.10]
pos 2: embed(Marie)  -> s2 -> [-0.20,  0.55,  0.09, -0.31]
pos 3: embed(admire) -> s3 -> [ 0.41,  0.27, -0.15,  0.62]
pos 4: embed(Jean)   -> s4 -> [ 0.08, -0.39,  0.71,  0.24]
```

## Step 9: project each to vocabulary scores

A linear layer maps 4 numbers to 7 numbers (one per French word). Position 1:

```
logits  = [-2.1, -1.8, -3.0, -2.5,  4.2,  0.7,  1.1]
             PAD  EOS  SOS  UNK  Marie admire Jean

softmax = [0.00, 0.00, 0.00, 0.00, 0.93, 0.03, 0.04]
```

The model puts 0.93 on `Marie`. The label says `Marie` (ID 4). Loss at this position is `-log(0.93) = 0.07`, small, because it was right.

Suppose position 3 goes badly:

```
softmax = [0.00, 0.01, 0.00, 0.02, 0.05, 0.12, 0.80]
label = Jean (ID 6), probability 0.80,  loss = -log(0.80) = 0.22
```

## Step 10: total loss, backward

Average the per-position losses, backpropagate. Gradients flow back through s4 to s1, through the handoff into h4, and then back through h3, h2, h1 into the encoder and the embedding tables. This is the only way the encoder learns anything: the signal arrives entirely through that one handoff.

## Now inference on the same sentence

Steps 1 to 7 identical. Step 8 becomes a loop, because no `dec_in` exists.

```
s = [0.51, 0.23, -0.18, 0.29]      from encoder
token = <SOS>

iteration 1:
  feed embed(<SOS>), state s
  softmax -> Marie at 0.93
  emit Marie,  s updates

iteration 2:
  feed embed(Marie)          <- its own output, not a label
  softmax -> admire at 0.88
  emit admire, s updates

iteration 3:
  feed embed(admire)
  softmax -> Jean at 0.80
  emit Jean, s updates

iteration 4:
  feed embed(Jean)
  softmax -> <EOS> at 0.91
  stop

output: Marie admire Jean
```

## The one difference, side by side

| | training | inference |
|---|---|---|
| decoder input at step 2 | true `Marie` from the label | whatever step 1 emitted |
| execution | one parallel call | Python loop |
| stopping | target length is known | when `<EOS>` appears |

If the model had emitted `Jean` at iteration 1, training would still have fed it `Marie` at iteration 2. Inference feeds it `Jean`, and the output derails from there. That gap is the exposure problem, and it's the reason beam search exists.