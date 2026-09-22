use candle_core::{Result, Tensor};
use candle_nn::{Embedding, Module, VarBuilder, embedding};

use crate::config::TransformerConfig;

pub struct InputEmbedding {
    token: Embedding,
    position: Embedding,
    max_sequence_length: usize,
}

impl InputEmbedding {
    pub fn new(config: &TransformerConfig, builder: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            token: embedding(
                config.vocab_size,
                config.model_width,
                builder.pp("token"),
            )?,
            position: embedding(
                config.max_sequence_length,
                config.model_width,
                builder.pp("position"),
            )?,
            max_sequence_length: config.max_sequence_length,
        })
    }

    pub fn forward(&self, token_ids: &Tensor) -> Result<Tensor> {
        let (batch_size, sequence_length) = token_ids.dims2()?;
        if sequence_length > self.max_sequence_length {
            candle_core::bail!(
                "sequence length {sequence_length} exceeds maximum {}",
                self.max_sequence_length
            )
        }

        let positions = Tensor::arange(0_u32, sequence_length as u32, token_ids.device())?
            .unsqueeze(0)?
            .expand((batch_size, sequence_length))?;
        self.token.forward(token_ids)? + self.position.forward(&positions)?
    }
}
