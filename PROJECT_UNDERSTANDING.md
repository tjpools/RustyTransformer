# Project Understanding: Tiny Decoder-Only Transformer

This project is a small, inference-only decoder Transformer written in Rust with
[Candle](https://github.com/huggingface/candle). Its purpose is to make the model's
data flow and tensor shapes easy to inspect, rather than to provide a complete
language-model training or text-generation system.

## 1. The Whole Forward Pass

The executable creates a tiny model and passes four integer token IDs through it:

```text
token IDs [B, T]
    |
    v
token embeddings + position embeddings [B, T, C]
    |
    v
decoder block 0 [B, T, C]
    |
    v
decoder block 1 [B, T, C]
    |
    v
final layer normalization [B, T, C]
    |
    v
language-model head [B, T, V]
```

The symbols used throughout this guide are:

| Symbol | Meaning | Tiny configuration |
| --- | --- | ---: |
| `B` | batch size | `1` in the example |
| `T` | sequence length | `4` in the example, at most `16` |
| `C` | model width | `24` |
| `H` | attention head count | `4` |
| `D` | width of one head, `C / H` | `6` |
| `F` | feed-forward hidden width | `48` |
| `V` | vocabulary size | `32` |

For the example input `[[1, 5, 9, 2]]`, the complete shape journey is:

```text
[1, 4] -> [1, 4, 24] -> [1, 4, 24] -> [1, 4, 24] -> [1, 4, 32]
```

## 2. Configuration and Invariants

Source: [`src/config.rs`](src/config.rs)

`TransformerConfig` defines all dimensions needed to construct the model. The
`tiny()` preset creates a model with two decoder blocks and four attention heads.

The important invariant is:

$$
C \bmod H = 0
$$

This lets each token's width be divided evenly among the heads. Here,
$D = 24 / 4 = 6$. `Transformer::new` validates this before constructing layers.

The configuration controls tensor dimensions; it does not hold learned values.
The learned values are created through Candle's `VarBuilder`.

## 3. Tensor Inspection

Source: [`src/tensor.rs`](src/tensor.rs)

`TensorDescriptor` is a debugging helper, not part of the model's mathematics. It
extracts three useful properties from a Candle tensor:

- dimensions, such as `[1, 4, 32]`
- data type, such as `F32` or `U32`
- device, such as `cpu`

The example uses it to show that integer token IDs become floating-point logits.
Candle itself owns tensor storage, device placement, broadcasting, and numerical
kernels; this project composes those operations into a Transformer.

## 4. Token and Position Embeddings

Source: [`src/model/embedding.rs`](src/model/embedding.rs)

The input is a rank-two integer tensor:

```text
token_ids: [B, T]
```

There are two learned lookup tables:

```text
token table:    [V, C] = [32, 24]
position table: [max_T, C] = [16, 24]
```

For every token ID, the token table supplies a vector of width `C`. The code also
creates positions `0..T`, expands them across the batch, and looks up one learned
position vector for each sequence location. The two vectors are added:

$$
x_{b,t} = E_{token}[id_{b,t}] + E_{position}[t]
$$

The output has shape `[B, T, C]`. Position embeddings matter because attention by
itself has no concept of token order. The method rejects sequences longer than
`max_sequence_length` before performing the lookups.

## 5. Multi-Head Causal Self-Attention

Source: [`src/model/attention.rs`](src/model/attention.rs)

Attention starts with hidden states of shape `[B, T, C]` and applies three learned
linear projections:

$$
Q = XW_Q + b_Q, \qquad K = XW_K + b_K, \qquad V = XW_V + b_V
$$

Each result still has shape `[B, T, C]`. `split_heads` then reshapes and transposes
each one:

```text
[B, T, C] -> [B, T, H, D] -> [B, H, T, D]
```

For this model that final shape is `[B, 4, T, 6]`.

### Attention scores

Each query is compared with every key in the same head:

$$
S = \frac{QK^T}{\sqrt{D}}
$$

The matrix multiplication has these shapes:

```text
Q:             [B, H, T, D]
transpose(K):  [B, H, D, T]
scores:        [B, H, T, T]
```

Dividing by $\sqrt{D}$ keeps dot products from growing too large as the head width
increases, which helps softmax avoid becoming excessively sharp.

### Causal mask

The mask has shape `[1, 1, T, T]`. It contains `0` on and below the diagonal and
negative infinity above it:

```text
T = 4

  0  -inf -inf -inf
  0    0   -inf -inf
  0    0    0   -inf
  0    0    0    0
```

It broadcasts across batches and heads. After adding the mask, softmax turns every
masked future position into probability zero. Thus position `t` can attend only to
positions `0..t`, which preserves autoregressive causality.

### Weighted values and merged heads

Softmax is applied across the last dimension:

$$
A = \operatorname{softmax}(S + M)
$$

The attention probabilities mix the value vectors:

$$
Z = AV
$$

```text
weights:  [B, H, T, T]
values:   [B, H, T, D]
attended: [B, H, T, D]
```

The code transposes, makes the tensor contiguous, and reshapes the heads back into
`[B, T, C]`. A final learned output projection lets information from the heads mix.

## 6. The Feed-Forward Network

Source: [`src/model/feed_forward.rs`](src/model/feed_forward.rs)

Attention mixes information between sequence positions. The feed-forward network
instead transforms each position independently, using the same learned function at
every position:

$$
\operatorname{FFN}(x) = W_2\operatorname{GELU}(W_1x + b_1) + b_2
$$

Its shape journey is:

```text
[B, T, C] -> [B, T, F] -> GELU -> [B, T, C]
[B, T, 24] -> [B, T, 48] -> GELU -> [B, T, 24]
```

The expansion gives each token more room for nonlinear feature transformation. The
contraction restores model width so the result can enter the residual stream.

## 7. One Pre-Norm Decoder Block

Source: [`src/model/block.rs`](src/model/block.rs)

A block combines attention and the feed-forward network with layer normalization
and residual connections:

$$
r = x + \operatorname{Attention}(\operatorname{LayerNorm}(x))
$$

$$
y = r + \operatorname{FFN}(\operatorname{LayerNorm}(r))
$$

This is called a **pre-norm** block because normalization occurs before each
sublayer. Both residual additions require their two inputs to have the same shape,
so the block preserves `[B, T, C]` from entrance to exit.

Residual connections maintain a direct path for existing information while each
sublayer contributes an update. Layer normalization operates across each token's
`C` features and helps control the scale of values entering a sublayer.

## 8. Block Stack and Language-Model Head

Source: [`src/model/transformer.rs`](src/model/transformer.rs)

`Transformer::new` creates `layer_count` independent blocks. The `VarBuilder`
prefixes (`block-0`, `block-1`, and so on) place each block's learned parameters in
a distinct namespace.

`Transformer::forward` performs three steps:

1. Convert `[B, T]` token IDs to `[B, T, C]` embeddings.
2. Pass the hidden states through every block in order.
3. Apply final layer normalization and project `C` features to `V` vocabulary scores.

The language-model head produces `[B, T, V]`. Every vector of length `V` contains
raw, unnormalized **logits** for the token that could follow that position's prefix:

```text
logits[b, t, token_id]
```

There is deliberately no final softmax. Training losses and sampling routines often
consume logits directly because they can apply numerically stable operations or
temperature scaling themselves.

## 9. Executable Wiring

Source: [`src/main.rs`](src/main.rs)

The binary:

1. selects the CPU device and `F32` model parameters;
2. creates a `VarMap` and a `VarBuilder`;
3. constructs the tiny model with newly initialized parameters;
4. creates one sequence containing four `U32` token IDs;
5. runs a forward pass and prints input and output descriptors.

The output values are not meaningful language predictions because the parameters
have not been trained. The executable demonstrates successful model construction,
shape flow, and numerical execution.

## 10. What the Tests Establish

Two focused unit tests currently define the core behavioral guarantees:

- `mask_blocks_future_positions` verifies that the causal mask leaves permitted
  entries at zero and places negative infinity on future entries.
- `forward_maps_tokens_to_vocabulary_logits` verifies that `[1, 4]` token IDs
  produce logits with shape `[1, 4, vocab_size]`.

Run them with:

```bash
cargo test
```

These tests establish causal masking and end-to-end shape compatibility. They do
not yet test exact attention values, sequence-length rejection, invalid configs,
multiple batches, or learned behavior.

## 11. What This Demo Does Not Include

Keeping these boundaries clear prevents confusing the model core with a complete
language-model application:

- no tokenizer that maps text to token IDs
- no dataset, loss function, backpropagation, optimizer, or training loop
- no checkpoint loading or saving
- no next-token sampler or autoregressive generation loop
- no padding mask for variable-length examples
- no dropout or training/evaluation mode distinction
- no key/value cache for efficient generation
- no rotary embeddings; positions use a learned lookup table
- no weight tying between token embeddings and the language-model head

The project implements the essential forward computation of a small decoder-only
Transformer. Those omitted pieces would turn it into a trainable or text-generating
system, but they are not required to study the architecture itself.

## 12. A Useful Reading and Experiment Order

1. Change values in `TransformerConfig::tiny` and predict every resulting shape.
2. Print `TensorDescriptor`s after embedding, attention, and each block.
3. Construct a small causal mask by hand and compare it with the unit test.
4. Trace one query row through score scaling, masking, softmax, and value mixing.
5. Add a test for an overlong sequence and one for a batch larger than one.
6. Add greedy next-token selection, while remembering that an untrained model's
   selected tokens will be arbitrary.

The central mental model is simple: embeddings create a residual stream of width
`C`; attention moves information across allowed positions; the feed-forward network
transforms information within each position; residual connections preserve and
accumulate updates; and the language-model head converts the final state at every
position into vocabulary logits.