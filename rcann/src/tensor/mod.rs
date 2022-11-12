mod dims;
mod native;

pub use dims::*;
pub use native::base::*;
pub use native::iter::*;
pub use native::tensor::*;
pub use native::view::*;
use std::fmt::Debug;

use crate::dtype::DType;

pub trait ITensorBase<T: DType>: Debug {
    fn len(&self) -> usize;
    fn dims(&self) -> &Dims;
}

pub trait ITensor<T: DType>: ITensorBase<T> {
    fn resize<D>(&mut self, dims: D)
    where
        D: Into<Dims>;
    fn resize_first_dim(&mut self, dim0: usize);
    fn copy_from_native(&mut self, native: &TensorView<T>);
    fn copy_to_native(&self, native: &mut TensorViewMut<T>);
}
