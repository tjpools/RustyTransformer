use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear};

use crate::config::TransformerConfig;

pub struct FeedForward {
    expand: Linear,
    contract: Linear,
}

impl FeedForward {
    pub fn new(config: &TransformerConfig, builder: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            expand: linear(
                config.model_width,
                config.feed_forward_width,
                builder.pp("expand"),
            )?,
            contract: linear(
                config.feed_forward_width,
                config.model_width,
                builder.pp("contract"),
            )?,
        })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.contract.forward(&self.expand.forward(input)?.gelu_erf()?)
    }
}
