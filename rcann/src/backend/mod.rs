use crate::dtype::DType;
use crate::tensor::{Dims, ITensor, TensorBase, TensorBaseMut};
use std::fmt::Debug;

mod cpu;

pub use cpu::*;

pub trait TensorTyped {
    type DType: DType;
    type Tensor: ITensor<Self::DType>;
}

pub trait TensorOps: TensorTyped {

    fn new_tensor<D>(&self, dim: D) -> Self::Tensor where D: Into<Dims>;
    fn resize_tensor<D>(&self, tensor: &mut Self::Tensor, dims: D) where D: Into<Dims>;
    fn write_tensor<T>(&self, tensor: &mut Self::Tensor, native_src: &T) where T: TensorBase<Self::DType>;
    fn read_tensor<T>(&self, tensor: &Self::Tensor, native_dst: &mut T) where T: TensorBaseMut<Self::DType>;

    fn resize_tensor_first_dim(&self, tensor: &mut Self::Tensor, first_dim_size: usize) {
        self.resize_tensor(tensor, tensor.dims().with_resized_first_axis(first_dim_size));
    }

    fn new_tensor_from_native<T>(&self, native: T) -> Self::Tensor where T: TensorBase<Self::DType> {
        let mut tensor = self.new_tensor(native.dims());
        self.write_tensor(&mut tensor, &native);
        tensor
    }

}

pub trait MatrixMultiplication: TensorTyped {
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
}

pub trait BackendOther: TensorTyped {

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

pub trait Backend: 'static + Debug + TensorTyped + TensorOps + MatrixMultiplication + BackendOther {}
