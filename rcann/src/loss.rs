use crate::backend::Backend;

#[derive(Copy, Clone, Default)]
pub enum LossFn {
    #[default]
    MSE,
}

impl LossFn {

    pub fn compute<B: Backend>(&self, backend: &B, output: &B::Tensor, expected: &B::Tensor, result: &mut B::Tensor, result_deriv: &mut B::Tensor) {
        match self {
            LossFn::MSE => backend.mean_squared_error(output, expected, result, result_deriv),
        }
    }

}
