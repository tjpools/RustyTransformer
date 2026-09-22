use candle_core::{Result, Tensor};
use candle_nn::{LayerNorm, Linear, Module, VarBuilder, layer_norm, linear};

use super::{block::TransformerBlock, embedding::InputEmbedding};
use crate::config::TransformerConfig;

pub struct Transformer {
    embedding: InputEmbedding,
    blocks: Vec<TransformerBlock>,
    final_norm: LayerNorm,
    language_model_head: Linear,
}

impl Transformer {
    pub fn new(config: &TransformerConfig, builder: VarBuilder<'_>) -> Result<Self> {
        config.validate()?;
        let blocks = (0..config.layer_count)
            .map(|index| TransformerBlock::new(config, builder.pp(format!("block-{index}"))))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            embedding: InputEmbedding::new(config, builder.pp("embedding"))?,
            blocks,
            final_norm: layer_norm(config.model_width, 1e-5, builder.pp("final_norm"))?,
            language_model_head: linear(
                config.model_width,
                config.vocab_size,
                builder.pp("language_model_head"),
            )?,
        })
    }

    pub fn forward(&self, token_ids: &Tensor) -> Result<Tensor> {
        let mut hidden = self.embedding.forward(token_ids)?;
        for block in &self.blocks {
            hidden = block.forward(&hidden)?;
        }
        self.language_model_head
            .forward(&self.final_norm.forward(&hidden)?)
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    use super::Transformer;
    use crate::config::TransformerConfig;

    #[test]
    fn forward_maps_tokens_to_vocabulary_logits() -> candle_core::Result<()> {
        let device = Device::Cpu;
        let config = TransformerConfig::tiny();
        let variables = VarMap::new();
        let builder = VarBuilder::from_varmap(&variables, DType::F32, &device);
        let model = Transformer::new(&config, builder)?;
        let tokens = Tensor::new(&[[1_u32, 2, 3, 4]], &device)?;

        let logits = model.forward(&tokens)?;

        assert_eq!(logits.dims(), &[1, 4, config.vocab_size]);
        Ok(())
    }
}
