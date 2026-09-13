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