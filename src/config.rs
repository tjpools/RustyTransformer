#[derive(Debug, Clone)]
pub struct TransformerConfig {
    pub vocab_size: usize,
    pub max_sequence_length: usize,
    pub model_width: usize,
    pub head_count: usize,
    pub layer_count: usize,
    pub feed_forward_width: usize,
}

impl TransformerConfig {
    pub fn tiny() -> Self {
        Self {
            vocab_size: 32,
            max_sequence_length: 16,
            model_width: 24,
            head_count: 4,
            layer_count: 2,
            feed_forward_width: 48,
        }
    }

    pub fn head_width(&self) -> usize {
        self.model_width / self.head_count
    }

    pub fn validate(&self) -> candle_core::Result<()> {
        if self.model_width % self.head_count != 0 {
            candle_core::bail!(
                "model width {} must be divisible by head count {}",
                self.model_width,
                self.head_count
            )
        }
        Ok(())
    }
}
