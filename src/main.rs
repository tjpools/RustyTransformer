use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use transformer::{Transformer, TransformerConfig, tensor::TensorDescriptor};

fn main() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let config = TransformerConfig::tiny();
    let variables = VarMap::new();
    let builder = VarBuilder::from_varmap(&variables, DType::F32, &device);
    let model = Transformer::new(&config, builder)?;

    let token_ids = Tensor::new(&[[1_u32, 5, 9, 2]], &device)?;
    let logits = model.forward(&token_ids)?;

    println!("input:  {:?}", TensorDescriptor::from_tensor(&token_ids));
    println!("logits: {:?}", TensorDescriptor::from_tensor(&logits));
    Ok(())
}