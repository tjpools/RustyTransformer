# Transformer

A small decoder-only transformer built with Rust and Candle. The source tree follows the model's conceptual structure so tensor shapes remain visible at each boundary.

```text
src/
|-- config.rs              Model dimensions and invariants
|-- tensor.rs              Tensor-shape inspection helpers
|-- model/
|   |-- embedding.rs       Token and position embeddings
|   |-- attention.rs       Multi-head causal self-attention
|   |-- feed_forward.rs    Per-token MLP
|   |-- block.rs           Pre-norm decoder block
|   `-- transformer.rs     Block stack and language-model head
|-- lib.rs                 Public library surface
`-- main.rs                Tiny CPU forward pass
```

Run the example and tests:

```bash
cargo run
cargo test
```

The initial model is intentionally small and inference-only. Candle owns storage and kernels; this project keeps shape transformations explicit so the path from token IDs to logits can be inspected.
