use crate::backend::Backend;
use crate::tensor::{Dim1, Dim2};

#[derive(Copy, Clone, Default)]
pub enum LossFn {
    #[default]
    MSE,
}

impl LossFn {
    pub fn compute<B: Backend>(
        &self,
        backend: &B,
        output: &B::Tensor<Dim2>,
        expected: B::TensorRef<'_, Dim2>,
        result: &mut B::Tensor<Dim1>,
        result_deriv: &mut B::Tensor<Dim2>,
    ) {
        match self {
            LossFn::MSE => backend.mean_squared_error(output, expected, result, result_deriv),
        }
    }
}
