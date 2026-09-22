use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use transformer::{Transformer, TransformerConfig, tensor::TensorDescriptor};

fn main() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let config = TransformerConfig::tiny();
    let model = match weights_path()? {
        Some(path) => Transformer::from_safetensors(&config, path, DType::F32, &device)?,
        None => {
            let variables = VarMap::new();
            let builder = VarBuilder::from_varmap(&variables, DType::F32, &device);
            Transformer::new(&config, builder)?
        }
    };

    let token_ids = Tensor::new(&[[1_u32, 5, 9, 2]], &device)?;
    let logits = model.forward(&token_ids)?;

    println!("input:  {:?}", TensorDescriptor::from_tensor(&token_ids));
    println!("logits: {:?}", TensorDescriptor::from_tensor(&logits));
    Ok(())
}

fn weights_path() -> candle_core::Result<Option<PathBuf>> {
    let mut args = std::env::args_os().skip(1);
    match (args.next(), args.next(), args.next()) {
        (None, None, None) => Ok(None),
        (Some(flag), Some(path), None) if flag == "--weights" => Ok(Some(path.into())),
        _ => candle_core::bail!("usage: transformer [--weights <checkpoint.safetensors>]"),
    }
}
