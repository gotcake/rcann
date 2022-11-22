use crate::dtype::DType;
use crate::tensor::{Dim1, Dim2, Dims, DimsMore, ITensor, Tensor, Tensor1, TensorBase, TensorBaseMut};
use std::fmt::Debug;
use std::process::Output;

mod cpu;

pub use cpu::*;

pub trait TensorTyped {
    type DType: DType;
    type Tensor<D: Dims>: ITensor<D>;
    type InputAdaptionBuff<D: Dims>;
    type OutputAdaptionBuff<D: Dims>;
}

pub trait TensorOps: TensorTyped {
    fn new_tensor_exact<D: Dims>(&self, dim: D) -> Self::Tensor<D>;
    fn new_tensor_batch_sized<D: DimsMore>(&self, inner_dims: D) -> Self::Tensor<D::More>;
    fn resize_tensor<D: Dims>(&self, tensor: &mut Self::Tensor<D>, dims: D);
    fn write_tensor<T, D>(&self, tensor: &mut Self::Tensor<D>, native_src: &T)
    where
        T: TensorBase<Self::DType, D>,
        D: Dims;
    fn read_tensor<T, D>(&self, tensor: &Self::Tensor<D>, native_dst: &mut T)
    where
        T: TensorBaseMut<Self::DType, D>,
        D: Dims;

    fn resize_tensor_major<D: Dims>(&self, tensor: &mut Self::Tensor<D>, size: usize) {
        self.resize_tensor(tensor, tensor.dims().resize_major(size));
    }

    fn new_tensor_from_native<T, D>(&self, native: T) -> Self::Tensor<D>
    where
        T: TensorBase<Self::DType, D>,
        D: Dims,
    {
        let mut tensor = self.new_tensor_exact(*native.dims());
        self.write_tensor(&mut tensor, &native);
        tensor
    }

    fn new_input_adaption_buff<D: DimsMore>(&self, inner_dims: D) -> Self::InputAdaptionBuff<D::More>;
    fn new_output_adaption_buff<D: DimsMore>(&self, inner_dims: D) -> Self::OutputAdaptionBuff<D::More>;
    fn adapt_input<'a, D: Dims>(
        &self,
        buff: &'a mut Self::InputAdaptionBuff<D>,
        input: &'a Tensor<Self::DType, D>,
    ) -> &'a Self::Tensor<D>;
    fn adapt_output<'a, D: Dims>(
        &self,
        buff: &'a mut Self::OutputAdaptionBuff<D>,
        output: &'a Self::Tensor<D>,
    ) -> &'a Tensor<Self::DType, D>;

    fn max_batch_size(&self) -> usize;
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
    );
}

pub trait BackendOther: TensorTyped {
    fn column_sum(&self, alpha: Self::DType, a: &Self::Tensor<Dim2>, beta: Self::DType, b: &mut Self::Tensor<Dim1>);

    fn add_assign<D: Dims>(&self, alpha: Self::DType, a: &Self::Tensor<D>, beta: Self::DType, b: &mut Self::Tensor<D>);

    /// computes the sigmoid function for all elements in a given tensor
    fn sigmoid(&self, activation: &Self::Tensor<Dim2>, output: &mut Self::Tensor<Dim2>);
    fn sigmoid_error(
        &self,
        output: &Self::Tensor<Dim2>,
        out_error: &Self::Tensor<Dim2>,
        result: &mut Self::Tensor<Dim2>,
    );

    /// computes the leaky ReLU function for all elements in a given tensor
    fn relu(&self, leak: Self::DType, activation: &Self::Tensor<Dim2>, output: &mut Self::Tensor<Dim2>);
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

    fn flush(&self);
    fn sync(&self);

    fn accum_confusion_matrix_multiclass(&self, output: &Self::Tensor<Dim2>, expected: &Self::Tensor<Dim2>, matrix: &mut Self::Tensor<Dim2>);

}

pub trait Backend: 'static + Debug + TensorTyped + TensorOps + MatrixMultiplication + BackendOther {}


pub trait PreparedDataset {
    type Batch;
    type Iter: Iterator<Item=Self::Batch>;
    fn iter_batches(&self) -> Self::Iter;
}
