# Sequence to Sequence Learning with Neural Networks

Sutskever, Vinyals, Le (2014). First pass reading notes.

---

## 1. The problem

Deep neural networks of the time required inputs and outputs of fixed dimensionality. That is fine for classification, where an image maps to one label, but it fails for tasks where both sides are variable length ordered token sequences: translation, question answering, speech.

"Mapping sequences to sequences" means the input is a variable length ordered list of tokens (an English sentence) and the output is also a variable length list (a French sentence), with no requirement that the two lengths match or be known in advance.

## 2. The method

Two separate LSTMs.

**Encoder.** Reads the source sequence one token at a time, updating its hidden state at each step. It produces no output. When it reaches the end of sentence marker, its final state is the fixed size representation of the whole input.

**Decoder.** Initialized with that representation as its starting state. Generates the target sequence one token at a time. At each step it outputs a probability distribution over the vocabulary, a token is chosen, and that token is fed back as input for the next step. It stops when it emits `<EOS>`, which is how the output length gets decided rather than being fixed in advance.

Both are 4 layer LSTMs with 1000 units per layer and 1000 dimensional word embeddings. Trained end to end with SGD. No phrase tables, no alignment models, no linguistic annotation.

### Dimensions

- **1000**: the width of a single state vector (one `h` or one `c`, in one layer).
- **8000**: the size of the complete handoff from encoder to decoder, being 4 layers x 2 states (hidden and cell) x 1000.

The number that matters is that 8000 is constant. It does not grow with sentence length. A six word sentence and a sixty word sentence both get 8000 floats.

## 3. Worked example of the forward pass

Source `Mary admires John`, target `Marie admire Jean`. The paper reverses the source, so the encoder reads it backwards.

**Encoder:**

| step | state in | word read | state out | kept? |
|---|---|---|---|---|
| 1 | h0 = zeros | John | h1 | discarded |
| 2 | h1 | admires | h2 | discarded |
| 3 | h2 | Mary | h3 | discarded |
| 4 | h3 | `<EOS>` | h4 | kept, this is v |

The intermediate states are not useless, they are the path by which information reaches v. What is lost is the ability to address them individually afterward.

**Decoder:**

| step | state in | word fed in | predicts | state out |
|---|---|---|---|---|
| 1 | s0 = v | `<SOS>` | Marie | s1 |
| 2 | s1 | Marie | admire | s2 |
| 3 | s2 | admire | Jean | s3 |
| 4 | s3 | Jean | `<EOS>` | stop |

`<SOS>` is a fixed start token, not model output. From step 2 onward the decoder feeds itself its own previous prediction.

## 4. The bottleneck

Look at the decoder's input column: `<SOS>`, Marie, admire, Jean. Every one is French. No English token appears anywhere in the decoder's computation. When the decoder must produce `Jean`, the only trace of the word `John` is whatever residue survives in v after three subsequent LSTM updates overwrote and remixed the state.

Two consequences follow.

**Word order must be encoded in v.** Consider `Mary admires John` and `John admires Mary`. Same words, different order, different correct translation. The decoder sees nothing but the vector. If both produced the same v, the decoder would be mathematically incapable of producing different outputs. So the encoder must produce genuinely different vectors. The abstract's claim that representations are sensitive to word order is not a bonus finding, it is a requirement for the architecture to function.

**Long sentences are the predicted weak point.** Capacity is 8000 floats regardless of input length, and the encoder writes into that space sequentially with each step remixing what is already there. In a sixty word sentence, the opening content has been overwritten by fifty nine subsequent updates. Degradation is the expected outcome.

## 5. Source reversal

The source sentence is reversed before encoding. The target is not.

Counting steps between reading a source word and emitting its translation, on the six step timeline:

| | unreversed | reversed |
|---|---|---|
| Mary to Marie | 3 | 1 |
| admires to admire | 3 | 3 |
| John to Jean | 3 | 5 |
| total | 9 | 9 |

The average distance is unchanged. What changes is the distribution: "all medium" becomes "some easy, some hard."

Why that helps. Gradients attenuate with distance, so in the unreversed case every correspondence the model needs to learn carries weak signal. Reversal creates at least one near trivial connection at the sentence start, which is learned quickly. Because the decoder conditions on its own previous outputs, getting the first word right is disproportionately valuable: a correct first token is a good input for predicting the second, and so on. Reversal buys a foothold.

Reported effect: BLEU 25.9 to 30.6, perplexity 5.8 to 4.7. The paper presents the short dependency story as a hypothesis and states it does not have a complete explanation.

## 6. Training versus inference

**Training (teacher forcing).** The true previous target token is fed in at each step, regardless of what the model predicted. If the model gets step 1 wrong, step 2 still receives the correct word. Training never has to recover from its own errors.

**Inference.** No reference exists, so the model's own prediction is fed back. One wrong token early corrupts everything downstream, because the decoder conditions on its own history. This is the exposure problem.

**Beam search** mitigates it. Rather than committing to the argmax at each step, keep the B highest scoring partial sequences alive, scored by summed log probabilities. A hypothesis that lost at step 1 can win overall once later evidence arrives, which greedy decoding could never discover. A hypothesis leaves the beam when it emits `<EOS>` and is set aside as a complete candidate. The paper reports that beam size 2 captures most of the gain. Note that beam search mitigates the exposure problem without fixing it: the model is still conditioning at test time on sequences it never saw in training.

## 7. Data

WMT'14 English to French. A 12M sentence pair subset, 348M French words and 304M English words. Evaluation on newstest2014, 3003 sentence pairs.

The format is two line aligned plain text files. Line n of one is the translation of line n of the other. No alignments, no parse trees, no phrase tables. This is what "minimal assumptions on the sequence structure" means concretely.

Preprocessing of one pair:

1. Tokenize: `["We", "must", "act", "now", "."]`
2. Apply vocabulary limits, 160k source and 80k target by frequency rank. Anything below the cut becomes `UNK`.
3. Append `<EOS>`, reverse the source.
4. Map to integer IDs, which index into the embedding table.

### On UNK

The criterion is frequency rank, not absence from the corpus. The tail is cut for two reasons: the output softmax runs over the entire target vocabulary at every step, so size is expensive, and a word appearing three times in 12M sentences receives three gradient updates and never learns a usable embedding.

The cut is by type, so if a word falls below rank 160,000 then every instance of it becomes `UNK` and the model never sees it once. `UNK` is a single symbol standing in for tens of thousands of distinct words, so `Reykjavik`, `photolithography` and a typo are indistinguishable to the model.

Two failure modes. On input, an unseen word arrives as `UNK` and its identity is gone. On output, if the correct next token is a rare proper noun, the model cannot emit it at all, since it is not in the output vocabulary. This is what the abstract means by BLEU being penalized on out of vocabulary words.

The vocabulary asymmetry (160k source, 80k target) reflects the asymmetry of cost: reading an unknown word gives one fuzzy input, generating one gives a guaranteed wrong output token, and the output side is where the softmax expense lives.

The field's eventual answer was to abandon words as the unit. Byte pair encoding (Sennrich et al., 2015) splits rare words into known subword fragments, which is why `UNK` no longer appears in modern systems.

State resets between pairs. Sentence 4,000,001 starts from h0 = 0 with no memory of sentence 4,000,000, which is the mechanical reason the model has no notion of document context.

## 8. Results

| system | BLEU |
|---|---|
| Phrase based SMT baseline | 33.3 |
| LSTM ensemble | 34.8 |
| SMT reranked by LSTM | 36.5 |

**Phrase based SMT** was the pre neural state of the art: a large table of phrase pairs with probabilities, learned from aligned parallel text, combined with a language model and a reordering model, searched by a decoder. A pipeline of separately trained components with hand designed structure. The contrast with a single end to end model is the point of the comparison.

**Reranking.** The SMT system emits its 1000 best candidates, an n best list. Its own top choice is frequently not the best translation in the list. The LSTM scores each candidate by how likely that French sentence is given the English input, this is combined with the SMT score, and the list is reordered. The LSTM only judges here, it never generates. The combination beats both systems alone because they make different kinds of errors: the LSTM rescues good translations the SMT ranked too low, while remaining confined to candidates the SMT considered fluent.

The significance of 34.8 over 33.3 is that it was the first time a pure neural system beat a mature phrase based system on a large scale translation task.

## 9. Where the gap lies

The vulnerable assumption is that an entire sentence of arbitrary length and complexity survives compression into one fixed size vector.

The abstract claims the LSTM did not have difficulty on long sentences, but that is a claim about the length distribution of their test set. The capacity argument says degradation should appear eventually, and later work confirmed that fixed vector encoders fall off past roughly thirty words.

The reversal trick can be read as a symptom. If flipping the input order is worth several BLEU points, the optimization is fragile in a way the architecture is not accounting for. The same information crosses the same bottleneck either way, only the gradient path lengths changed.

On paragraphs: the architecture is length agnostic, since nothing in it knows what a sentence is, but the representation is not. A paragraph gets the same 8000 floats as a sentence. Document level translation with a fixed vector is possible in principle and poor in practice.

## 10. What came next

Bahdanau, Cho, Bengio (2015) attacked exactly this assumption with attention: keep every encoder state instead of only the last, and let the decoder compute a weighted read over all of them at each step.

| | Seq2Seq | with attention |
|---|---|---|
| encoder states kept | only the last | all |
| size of what the decoder reads | fixed, 8000 floats | grows with source length |
| decoder access to source word 1 | through the bottleneck only | direct |

When the decoder is about to emit `Jean`, it can place most of its weight on the encoder state for `John` and read from it directly. No bottleneck to survive, and no need for the reversal trick. That line runs to the Transformer.

---

## One line summary

A fixed size vector is both the whole idea and the whole limitation. Everything else in the surrounding literature, reversal and beam search and reranking and attention, is either working around that constraint or removing it.

## Open questions for a second pass

- What exactly does the ensembling contribute, and how much of 34.8 is the ensemble versus a single model.
- The details of the SMT rescoring setup: how the LSTM score is combined with the baseline score, and with what weighting.
- Their evidence on long sentences, specifically the length distribution of newstest2014 and what the per length breakdown actually shows.
- Whether reversal still helps once optimization is improved by other means, which would separate a real architectural fix from a training artifact.