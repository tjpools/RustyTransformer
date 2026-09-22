use candle_core::{Result, Tensor};
use candle_nn::{LayerNorm, Module, VarBuilder, layer_norm};

use super::{attention::CausalSelfAttention, feed_forward::FeedForward};
use crate::config::TransformerConfig;

pub struct TransformerBlock {
    attention_norm: LayerNorm,
    attention: CausalSelfAttention,
    feed_forward_norm: LayerNorm,
    feed_forward: FeedForward,
}

impl TransformerBlock {
    pub fn new(config: &TransformerConfig, builder: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            attention_norm: layer_norm(config.model_width, 1e-5, builder.pp("attention_norm"))?,
            attention: CausalSelfAttention::new(config, builder.pp("attention"))?,
            feed_forward_norm: layer_norm(
                config.model_width,
                1e-5,
                builder.pp("feed_forward_norm"),
            )?,
            feed_forward: FeedForward::new(config, builder.pp("feed_forward"))?,
        })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let attention_input = self.attention_norm.forward(input)?;
        let residual = (input + self.attention.forward(&attention_input)?)?;
        let feed_forward_input = self.feed_forward_norm.forward(&residual)?;
        residual + self.feed_forward.forward(&feed_forward_input)?
    }
}
