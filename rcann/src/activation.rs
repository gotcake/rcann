use crate::backend::Backend;
use crate::dtype::DType;
use crate::tensor::Dim2;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ActivationFn {
    #[default]
    Sigmoid,
    ReLU {
        leak: f64,
    },
    Softmax,
}

impl ActivationFn {
    pub fn compute<B: Backend>(&self, backend: &B, activation: &B::Tensor<Dim2>, output: &mut B::Tensor<Dim2>) {
        match self {
            ActivationFn::Sigmoid => backend.sigmoid(activation, output),
            &ActivationFn::ReLU { leak } => backend.relu(B::DType::from_f64(leak), activation, output),
            ActivationFn::Softmax => backend.softmax(activation, output),
        }
    }

    pub fn compute_error<B: Backend>(
        &self,
        backend: &B,
        activation: &B::Tensor<Dim2>,
        output: &B::Tensor<Dim2>,
        out_error: &B::Tensor<Dim2>,
        result: &mut B::Tensor<Dim2>,
    ) {
        match self {
            ActivationFn::Sigmoid => backend.sigmoid_error(output, out_error, result),
            &ActivationFn::ReLU { leak } => backend.relu_error(B::DType::from_f64(leak), activation, out_error, result),
            ActivationFn::Softmax => backend.softmax_error(output, out_error, result),
        }
    }
}
