use crate::dtype::DType;
use crate::tensor::{Dim1, Dim2, Dims, ITensor, TensorBase, TensorBaseMut};
use std::fmt::Debug;

mod cpu;

pub use cpu::*;

pub trait TensorTyped {
    type DType: DType;
    type Tensor<D: Dims>: ITensor<Self::DType, D>;
}

pub trait TensorOps: TensorTyped {
    fn new_tensor<D>(&self, dim: D) -> Self::Tensor<D>
    where
        D: Dims;
    fn resize_tensor<D>(&self, tensor: &mut Self::Tensor<D>, dims: D)
    where
        D: Dims;
    fn write_tensor<T, D>(&self, tensor: &mut Self::Tensor<D>, native_src: &T)
    where
        T: TensorBase<Self::DType, D>,
        D: Dims;
    fn read_tensor<T, D>(&self, tensor: &Self::Tensor<D>, native_dst: &mut T)
    where
        T: TensorBaseMut<Self::DType, D>,
        D: Dims;

    fn resize_tensor_first_dim<D>(&self, tensor: &mut Self::Tensor<D>, first_dim_size: usize)
    where
        D: Dims,
    {
        self.resize_tensor(
            tensor,
            tensor.dims().with_resized_first_axis(first_dim_size),
        );
    }

    fn new_tensor_from_native<T, D>(&self, native: T) -> Self::Tensor<D>
    where
        T: TensorBase<Self::DType, D>,
        D: Dims,
    {
        let mut tensor = self.new_tensor(*native.dims());
        self.write_tensor(&mut tensor, &native);
        tensor
    }
}

pub trait MatrixMultiplication: TensorTyped {
    /// performs a generic matrix multiplication (gemm) operation
    fn matmul(
        &self,
        alpha: Self::DType,
        a: &Self::Tensor<Dim2>,
        ta: bool,
        b: &Self::Tensor<Dim2>,
        tb: bool,
        beta: Self::DType,
        c: &mut Self::Tensor<Dim2>,
        tc: bool,
    );
}

pub trait BackendOther: TensorTyped {
    fn column_sum(
        &self,
        alpha: Self::DType,
        a: &Self::Tensor<Dim2>,
        beta: Self::DType,
        b: &mut Self::Tensor<Dim1>,
    );

    fn add_assign<D>(
        &self,
        alpha: Self::DType,
        a: &Self::Tensor<D>,
        beta: Self::DType,
        b: &mut Self::Tensor<D>,
    ) where
        D: Dims;

    /// computes the sigmoid function for all elements in a given tensor
    fn sigmoid(&self, activation: &Self::Tensor<Dim2>, output: &mut Self::Tensor<Dim2>);
    fn sigmoid_error(
        &self,
        output: &Self::Tensor<Dim2>,
        out_error: &Self::Tensor<Dim2>,
        result: &mut Self::Tensor<Dim2>,
    );

    /// computes the leaky ReLU function for all elements in a given tensor
    fn relu(
        &self,
        leak: Self::DType,
        activation: &Self::Tensor<Dim2>,
        output: &mut Self::Tensor<Dim2>,
    );
    fn relu_error(
        &self,
        leak: Self::DType,
        activation: &Self::Tensor<Dim2>,
        out_error: &Self::Tensor<Dim2>,
        result: &mut Self::Tensor<Dim2>,
    );

    fn softmax(&self, activation: &Self::Tensor<Dim2>, output: &mut Self::Tensor<Dim2>);
    fn softmax_error(
        &self,
        output: &Self::Tensor<Dim2>,
        out_error: &Self::Tensor<Dim2>,
        result: &mut Self::Tensor<Dim2>,
    );

    fn mean_squared_error(
        &self,
        output: &Self::Tensor<Dim2>,
        expected: &Self::Tensor<Dim2>,
        result: &mut Self::Tensor<Dim1>,
        result_deriv: &mut Self::Tensor<Dim2>,
    );
}

pub trait Backend:
    'static + Debug + TensorTyped + TensorOps + MatrixMultiplication + BackendOther
{
}
