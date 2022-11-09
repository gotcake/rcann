use crate::dtype::DType;
use crate::tensor::{Tensor, Tensor2};

pub trait Backend {

    type DType: DType;
    type Tensor: Tensor<Self::DType>;
    type Tensor2: Tensor2<Self::DType>;

    fn new_tensor1(&self, len: usize) -> Self::Tensor;
    fn new_tensor2(&self, rows: usize, cols: usize) -> Self::Tensor2;
    fn new_tensor_from(&self, data: Vec<Self::DType>) -> Self::Tensor;
    fn new_tensor2_from(&self, rows: usize, cols: usize, data: Vec<Self::DType>) -> Self::Tensor2;
    fn resize_tensor1(&self, tensor: &mut Self::Tensor, new_len: usize);
    fn resize_tensor2(&self, tensor: &mut Self::Tensor2, new_rows: usize, new_cols: usize);

    /// performs a generic matrix multiplication (gemm) operation
    fn matmul(
        &self,
        alpha: Self::DType,
        a: &Self::Tensor2,
        ta: bool,
        b: &Self::Tensor2,
        tb: bool,
        beta: Self::DType,
        c: &mut Self::Tensor2,
        tc: bool,
    );

    fn column_sum(&self, alpha: Self::DType, a: &Self::Tensor2, beta: Self::DType, b: &mut Self::Tensor);

    fn add_assign(&self, alpha: Self::DType, a: &Self::Tensor, beta: Self::DType, b: &mut Self::Tensor);
    fn add_assign2(&self, alpha: Self::DType, a: &Self::Tensor2, beta: Self::DType, b: &mut Self::Tensor2);

    /// computes the sigmoid function for all elements in a given tensor
    fn sigmoid(&self, activation: &Self::Tensor2, output: &mut Self::Tensor2);
    fn sigmoid_error(&self, output: &Self::Tensor2, out_error: &Self::Tensor2, result: &mut Self::Tensor2);

    /// computes the leaky ReLU function for all elements in a given tensor
    fn relu(&self, leak: Self::DType, activation: &Self::Tensor2, output: &mut Self::Tensor2, );
    fn relu_error(&self, leak: Self::DType, activation: &Self::Tensor2, out_error: &Self::Tensor2, result: &mut Self::Tensor2);

    fn softmax(&self, activation: &Self::Tensor2, output: &mut Self::Tensor2);
    fn softmax_error(&self, output: &Self::Tensor2, out_error: &Self::Tensor2, result: &mut Self::Tensor2);

}