# Implementing Sutskever et al. (2014): Sequence to Sequence Learning with Neural Networks

Paper: [arXiv:1409.3215](https://arxiv.org/abs/1409.3215)

## Quick start

```bash
uv sync
python3 seq2seq.py
```

## Goal

Implement the paper from scratch in PyTorch: the stacked LSTM encoder-decoder, its training procedure, source reversal, and beam search decoding. The goal was to implement the method faithfully and understand every component, not to match the paper's BLEU score, which would require its data and compute.

## The synthetic task

Instead of WMT'14 English to French, the model is trained on a synthetic translation task.

- Each source sentence is 2 to 5 adjective-noun phrases, for example `red house`.
- The target translates each word and swaps the order to noun then adjective, as in French, for example `la maison rouge`.
- Some nouns carry an article (`le`, `la`) and some do not, so source and target lengths differ.
- There is exactly one correct translation per input.
- 6000 training pairs and 300 held-out pairs, generated from separately seeded random generators.
- Vocabulary: 16 source words and 18 target words, plus 4 special tokens (`<pad>`, `<sos>`, `<eos>`, `<unk>`).

This keeps the parts of the problem the paper cares about (word translation, local reordering, variable length) while making it small enough to train on a CPU.

## What follows the paper

| Component | Paper | This implementation |
|---|---|---|
| Encoder / decoder | Separate 4-layer LSTMs | Same |
| Embedding size | Equal to hidden size | Same |
| Context | Final encoder state only, no attention | Same |
| Initialization | Uniform in [-0.08, 0.08] | Same |
| Optimizer | SGD without momentum | Same |
| Batch size | 128 | Same (baseline) |
| Gradient scaling | Divided by number of sentences | Same |
| Gradient clipping | Rescale to norm 5 if norm exceeds 5 | Same |
| Learning rate schedule | Flat, then 5 halvings | Same shape |
| Batching | Sentences of similar length grouped together | Same |
| Source reversal | Reverse source, keep target order | Same |
| Decoding | Left-to-right beam search, small beam | Beam width 2 |

## Deviations

| | Paper | This implementation | Reason |
|---|---|---|---|
| Hidden / embedding size | 1000 | 128 | Compute |
| Parameters | about 384M | about 1.06M | Compute |
| Data | about 12M sentence pairs | 6000 synthetic pairs | Compute and data |
| Vocabulary | 160k source / 80k target | 16 / 18 words | Synthetic task |
| Learning rate | 0.7 | 1.0 (baseline) | Initial choice, later changed to 0.7 |
| Schedule | 5 flat epochs, halve every half epoch, 7.5 epochs total | 35 flat epochs, halve once per epoch, 40 total | Stretched to the smaller dataset |
| Metric | BLEU | Exact match | One correct answer per input, so partial credit hides whether the mapping was learned |
| Ensembles | 5 models | 1 model | Compute |

## Pipeline overview

1. **Data generation.** Seeded generators produce training and validation pairs. Validation uses its own seed, so it does not change when the training set size changes.
2. **Vocabularies.** One per language, mapping words to integer ids, with reserved ids for the special tokens.
3. **Tensorization.** The source is reversed and then `<eos>` is appended, so `<eos>` is the last token the encoder reads. The target is wrapped in `<sos>` and `<eos>`.
4. **Batching.** Examples are sorted by source length, cut into batches, padded, and the batch order is shuffled each epoch. True source lengths are kept for packing.
5. **Model.** The encoder's final `(h, c)` initializes the decoder. A linear projection maps each decoder state to scores over the target vocabulary.
6. **Training.** Teacher forcing: the decoder input is the target shifted right, the labels are the target shifted left. Loss is summed over real tokens and divided by the number of sentences.
7. **Decoding.** Greedy decoding and beam search, both starting from `<sos>` and stopping at `<eos>` or a length cap.
8. **Evaluation.** Exact match on held-out pairs, plus two sanity checks: a shape check on the encoder-decoder handoff and an overfit check on 20 examples.

## Bugs found during implementation

| Bug | Symptom | Fix |
|---|---|---|
| Typo `cfg.max_pharses` | `AttributeError` when building the dataset | Renamed to `cfg.max_phrases` |
| Duplicate tokens in the target vocabulary (`la` three times, `le` twice) | Vocabulary of 25 instead of 22, unused ids | `sorted(set(tokens) - set(SPECIALS))` |
| Missing `random`, `torch` and `math` imports | `NameError` | Added the imports |
| Encoder LSTM missing `batch_first=True` | No error, but the encoder read across the batch as if it were time | Added `batch_first=True` |
| Right padding after `<eos>` | Encoder read padding, so its final state was not the state at `<eos>` | Kept true lengths and used `pack_padded_sequence` |
| `nn.init.uniform` | Deprecated function | `nn.init.uniform_` |
| Initialization overwrote the `<pad>` embedding rows | Padding mapped to a random fixed vector | Zeroed the rows again after initialization |
| Batch shuffle used the global `random` | Batch order not reproducible | Seeded generator passed into `make_batches` |
| `rng` used in `make_batches` without being a parameter | `UnboundLocalError`, because `rng` was assigned later in the function and so treated as local | Added `rng=None` to the signature |
| Generator recreated on every call | Every epoch would get the identical batch order | Created once in `train` and passed in |
| Beam search: finished hypotheses used up beam slots | Beam shrank toward greedy search | Keep collecting until there are B live beams |
| Beam search: no early stop | Ran to the length cap unnecessarily | Stop when the best finished score beats every live beam, since scores only decrease |
| Beam search: live beams dropped at the length cap | Higher-scoring unfinished beams ignored | Include them as final candidates |
| `argmax(..., keeptime=True)` | `TypeError` | `keepdim=True` |
| Old function signatures after adding source lengths | `TypeError` in several call sites | Passed `src_lengths` through the model, loss, decoders and checks |

The most dangerous of these were the silent ones: the missing `batch_first=True`, the padding read by the encoder, and the repeated batch order. None of them raised an error, and each would have made the results wrong without any obvious sign.

## Results

### Run 1: baseline (lr 1.0, batch 128, halving every epoch at the end)

- Overfit check (20 pairs, 600 steps): loss per token fell from 3.08 to 0.32, but only 20% exact match.
- Full training: loss per token reached about 1.44 by epoch 15 and stayed there for the rest of training.
- Held-out exact match: 0.0% greedy, 0.0% beam.

### Run 2: lr 0.7, batch 32, halvings spread over the last third

- Overfit check: loss per token 0.31, 10% exact match.
- Full training: the same plateau at about 1.44, reached by epoch 5.
- Held-out exact match: 0.0% greedy, 0.0% beam.

### Sample predictions (run 2)

```
source      red house red cat new stone small house
reference   la maison rouge le chat rouge la pierre neuf la maison petit
predicted   la pierre petit la riviere petit la maison vert la riviere vert

source      red road small cat green field cold bird
reference   route rouge le chat petit le champ vert oiseau froid
predicted   la pierre petit la riviere petit la maison vert la riviere vert
```

## Analysis

### The decoder learned the target language but ignored the source

Three observations point to the same conclusion:

1. **Different inputs give nearly identical outputs.** A model reading the source would produce different translations for different sentences.
2. **The outputs have the right structure but the wrong content.** Article, noun, adjective in the right order, with the right article attached to each noun. That is the pattern of the target language, learned without reference to the input.
3. **The plateau matches an unconditional language model.** With 8 nouns and 8 adjectives chosen at random and the articles determined by the noun, a model that guesses the content from the target side alone has a loss of roughly 1.4 to 1.5 per token. The observed plateau is 1.44.

In effect, the decoder became a language model over the target side. This is a known failure mode for encoder-decoder models: learning the target distribution is easy, and learning to use the source is harder, so the model settles on the easy solution first.

### It is not a wiring bug

In the overfit check the loss reached about 0.31, well below the 1.44 plateau. A model cannot get below that level without using the source, so information does flow from encoder to decoder. The architecture is connected correctly.

The gap between a loss of 0.31 and an exact match of 10 to 20% is also consistent. At that loss the model gets roughly 90% of individual tokens right, and with about 13 target tokens per sentence the chance of getting all of them right is about 0.9 to the power 13, around 25%.

### It is not the learning rate or the number of updates alone

Run 2 used the paper's learning rate, four times as many updates per epoch, and a longer decay phase. The plateau was reached sooner and did not move. So the model is not short of steps. Something makes it hard for the learning signal to reach the encoder at all.

### Leading hypothesis: the initialization range does not transfer to a narrower model

The paper's initialization of [-0.08, 0.08] was chosen for 1000-unit layers. Each LSTM unit sums contributions from all its inputs, so with 1000 units the initial signals add up to a useful size. With 128 units the same range produces much weaker signals and gradients. Stacked through four encoder layers, the handoff, and four decoder layers, the signal reaching the encoder may be too small for it to learn a useful representation. The decoder can still reduce the loss on its own, so it does.

This is a hypothesis, not a finding. It has not yet been measured.

## Next steps

1. **Measure before changing anything.** A diagnostic that compares the loss with the real encoder state, a zeroed state, and another sentence's state, and that compares the gradient norms of the encoder and decoder. Equal losses would confirm the source is ignored. A very small encoder-to-decoder gradient ratio would confirm the encoder is starved of learning signal.
2. **If the encoder is starved,** try one change at a time: a larger initialization range scaled to the hidden size, fewer layers, or a forget-gate bias of 1.0 (a trick from after the paper that helps gradients flow through LSTMs).
3. **Once the model learns the task,** run the paper's key ablation with source reversal turned off and compare.

## Lessons

- **Implementing a method and reproducing a result are different things.** The architecture and training procedure follow the paper. The result does not transfer at this scale.
- **Hyperparameters carry assumptions about scale.** The initialization range, learning rate and schedule were tuned for 1000-unit layers trained on millions of sentences. Shrinking the model and data without adjusting them changes how they behave.
- **Silent bugs are the expensive ones.** The crashes were quick to fix. The bugs that ran without errors, such as the encoder reading across the batch and reading padding, would have been hard to find from the results alone.
- **Sanity checks earn their place.** The overfit check is what ruled out a wiring bug and pointed the investigation toward optimization.
- **Look at the outputs, not just the metric.** A score of 0% says only that the model failed. The sample predictions showed how it failed, which is what made the analysis possible.

## Citations

```bibtex
@misc{sutskever2014sequence,
    title   = {Sequence to Sequence Learning with Neural Networks},
    author  = {Ilya Sutskever and Oriol Vinyals and Quoc V. Le},
    year    = {2014},
    eprint  = {1409.3215},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL},
    url     = {https://arxiv.org/abs/1409.3215}
}
```