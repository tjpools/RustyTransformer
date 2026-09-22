use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear, ops::softmax_last_dim};

use crate::config::TransformerConfig;

pub struct CausalSelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    head_count: usize,
    head_width: usize,
}

impl CausalSelfAttention {
    pub fn new(config: &TransformerConfig, builder: VarBuilder<'_>) -> Result<Self> {
        let width = config.model_width;
        Ok(Self {
            query: linear(width, width, builder.pp("query"))?,
            key: linear(width, width, builder.pp("key"))?,
            value: linear(width, width, builder.pp("value"))?,
            output: linear(width, width, builder.pp("output"))?,
            head_count: config.head_count,
            head_width: config.head_width(),
        })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (batch_size, sequence_length, model_width) = input.dims3()?;
        let query = self.split_heads(self.query.forward(input)?)?;
        let key = self.split_heads(self.key.forward(input)?)?;
        let value = self.split_heads(self.value.forward(input)?)?;

        let scale = 1.0 / (self.head_width as f64).sqrt();
        let scores = query.matmul(&key.transpose(2, 3)?)?.affine(scale, 0.0)?;
        let mask = causal_mask(sequence_length, input.device())?;
        let weights = softmax_last_dim(&scores.broadcast_add(&mask)?)?;
        let attended = weights
            .matmul(&value)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch_size, sequence_length, model_width))?;

        self.output.forward(&attended)
    }

    fn split_heads(&self, tensor: Tensor) -> Result<Tensor> {
        let (batch_size, sequence_length, _) = tensor.dims3()?;
        tensor
            .reshape((
                batch_size,
                sequence_length,
                self.head_count,
                self.head_width,
            ))?
            .transpose(1, 2)
    }
}

fn causal_mask(sequence_length: usize, device: &candle_core::Device) -> Result<Tensor> {
    let mut values = vec![0_f32; sequence_length * sequence_length];
    for row in 0..sequence_length {
        for column in (row + 1)..sequence_length {
            values[row * sequence_length + column] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(values, (1, 1, sequence_length, sequence_length), device)
}

#[cfg(test)]
mod tests {
    use super::causal_mask;
    use candle_core::Device;

    #[test]
    fn mask_blocks_future_positions() -> candle_core::Result<()> {
        let values = causal_mask(3, &Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(values[0], 0.0);
        assert_eq!(values[3], 0.0);
        assert!(values[1].is_infinite() && values[1].is_sign_negative());
        assert!(values[5].is_infinite() && values[5].is_sign_negative());
        Ok(())
    }
}
