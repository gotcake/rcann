use crate::dtype::DType;
use crate::tensor::{Dims, ITensor, TensorBase};
use std::fmt::Debug;

mod cpu;

pub use cpu::CpuBackend;

pub trait Backend: 'static + Debug {
    type DType: DType;
    type Tensor: ITensor<Self::DType>;

    fn new_tensor<D: Into<Dims>>(&self, dim: D) -> Self::Tensor;
    fn new_tensor_from_native<T>(&self, native: T) -> Self::Tensor
    where
        T: TensorBase<Self::DType>;

    /// performs a generic matrix multiplication (gemm) operation
    fn matmul(
        &self,
        alpha: Self::DType,
        a: &Self::Tensor,
        ta: bool,
        b: &Self::Tensor,
        tb: bool,
        beta: Self::DType,
        c: &mut Self::Tensor,
        tc: bool,
    );

    fn column_sum(
        &self,
        alpha: Self::DType,
        a: &Self::Tensor,
        beta: Self::DType,
        b: &mut Self::Tensor,
    );
    fn add_assign(
        &self,
        alpha: Self::DType,
        a: &Self::Tensor,
        beta: Self::DType,
        b: &mut Self::Tensor,
    );

    /// computes the sigmoid function for all elements in a given tensor
    fn sigmoid(&self, activation: &Self::Tensor, output: &mut Self::Tensor);
    fn sigmoid_error(
        &self,
        output: &Self::Tensor,
        out_error: &Self::Tensor,
        result: &mut Self::Tensor,
    );

    /// computes the leaky ReLU function for all elements in a given tensor
    fn relu(&self, leak: Self::DType, activation: &Self::Tensor, output: &mut Self::Tensor);
    fn relu_error(
        &self,
        leak: Self::DType,
        activation: &Self::Tensor,
        out_error: &Self::Tensor,
        result: &mut Self::Tensor,
    );

    fn softmax(&self, activation: &Self::Tensor, output: &mut Self::Tensor);
    fn softmax_error(
        &self,
        output: &Self::Tensor,
        out_error: &Self::Tensor,
        result: &mut Self::Tensor,
    );

    fn mean_squared_error(
        &self,
        output: &Self::Tensor,
        expected: &Self::Tensor,
        result: &mut Self::Tensor,
        result_deriv: &mut Self::Tensor,
    );
}
