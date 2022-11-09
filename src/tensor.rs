use crate::dtype::DType;

pub trait Tensor<T: DType> {
    fn len(&self) -> usize;
}

pub trait Tensor2<T: DType> : Tensor<T> {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn dim(&self) -> (usize, usize);
}

