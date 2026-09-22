use candle_core::{DType, Device, Tensor, backend::BackendDevice};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorDescriptor {
    pub dimensions: Vec<usize>,
    pub dtype: DType,
    pub device: String,
}

impl TensorDescriptor {
    pub fn from_tensor(tensor: &Tensor) -> Self {
        Self {
            dimensions: tensor.dims().to_vec(),
            dtype: tensor.dtype(),
            device: device_name(tensor.device()),
        }
    }
}

fn device_name(device: &Device) -> String {
    match device {
        Device::Cpu => "cpu".to_owned(),
        Device::Cuda(cuda) => format!("cuda:{:?}", cuda.location()),
        Device::Metal(metal) => format!("metal:{:?}", metal.location()),
    }
}
