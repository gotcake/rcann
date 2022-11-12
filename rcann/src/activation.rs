use crate::backend::Backend;
use crate::dtype::DType;
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, Serialize, Deserialize, Default, PartialEq)]
pub enum ActivationFn {
    #[default]
    Sigmoid,
    ReLU {
        leak: f64,
    },
    Softmax,
}

impl ActivationFn {
    pub fn compute<B: Backend>(&self, backend: &B, activation: &B::Tensor, output: &mut B::Tensor) {
        match self {
            ActivationFn::Sigmoid => backend.sigmoid(activation, output),
            &ActivationFn::ReLU { leak } => {
                backend.relu(B::DType::from_f64(leak), activation, output)
            }
            ActivationFn::Softmax => backend.softmax(activation, output),
        }
    }

    pub fn compute_error<B: Backend>(
        &self,
        backend: &B,
        activation: &B::Tensor,
        output: &B::Tensor,
        out_error: &B::Tensor,
        result: &mut B::Tensor,
    ) {
        match self {
            ActivationFn::Sigmoid => backend.sigmoid_error(output, out_error, result),
            &ActivationFn::ReLU { leak } => {
                backend.relu_error(B::DType::from_f64(leak), activation, out_error, result)
            }
            ActivationFn::Softmax => backend.softmax_error(output, out_error, result),
        }
    }
}
