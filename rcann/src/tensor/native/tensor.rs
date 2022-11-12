use std::fmt::{Display, Debug, Formatter, Write};
use std::ops::{Deref, DerefMut};
use std::slice::{Iter, IterMut};
use std::vec::IntoIter;
use serde::Serializer;
use crate::dtype::DType;
use crate::impl_tensor_debug;
use crate::tensor::{ITensor, ITensorBase, TensorBase, TensorBaseMut, Dims, TensorView, TensorViewMut};

pub struct Tensor<T> {
    data: Vec<T>,
    dims: Dims,
}

impl<T> Tensor<T> {
    pub fn empty() -> Tensor<T> {
        Tensor {
            data: Vec::new(),
            dims: Dims::D1(0)
        }
    }
    pub fn from_vec<D: Into<Dims>>(data: Vec<T>, dim: D) -> Tensor<T> {
        let dim = dim.into();
        assert_eq!(data.len(), dim.tensor_len(), "Mismatched data length {} and dimension {:?}", data.len(), dim);
        Tensor {
            data,
            dims: dim,
        }
    }
    #[inline]
    pub(super) unsafe fn from_vec_unchecked(data: Vec<T>, dim: Dims) -> Tensor<T> {
        debug_assert_eq!(data.len(), dim.tensor_len());
        Tensor {
            data,
            dims: dim,
        }
    }
}

impl<T: DType> Tensor<T> {

    pub fn zero<D: Into<Dims>>(dim: D) -> Tensor<T> {
        let dim = dim.into();
        Tensor {
            data: vec![T::zero(); dim.tensor_len()],
            dims: dim,
        }
    }

    pub fn resize_fill<D: Into<Dims>>(&mut self, dim: D, fill: T) {
        let dim = dim.into();
        if self.dims != dim {
            let len = self.data.len();
            let new_len = dim.tensor_len();
            if new_len > self.data.capacity() {
                self.data.reserve_exact(new_len - len);
            }
            unsafe { self.data.set_len(new_len); }
            if new_len > len {
                self.data[len..new_len].fill(fill);
            }
            self.dims = dim;
        }
    }

}

impl<T: DType> ITensorBase<T> for Tensor<T> {
    #[inline]
    fn len(&self) -> usize {
        self.data.len()
    }
    #[inline]
    fn dims(&self) -> &Dims {
        &self.dims
    }
}

impl<T: DType> ITensor<T> for Tensor<T> {
    #[inline]
    fn resize<D>(&mut self, dims: D) where D: Into<Dims> {
        self.resize_fill(dims, T::ZERO);
    }
    #[inline]
    fn resize_first_dim(&mut self, dim0: usize) {
        self.resize_fill(self.dims.with_resized_first_axis(dim0), T::ZERO);
    }

    fn copy_from_native(&mut self, native: &TensorView<T>){
        assert_eq!(&self.dims, native.dims());
        self.data.copy_from_slice(native);
    }
    fn copy_to_native(&self, native: &mut TensorViewMut<T>) {
        assert_eq!(&self.dims, native.dims());
        native.copy_from_slice(&self.data);
    }
}

impl<T> Deref for Tensor<T> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T: DType> TensorBase<T> for Tensor<T> {
    #[inline]
    fn is_owned(&self) -> bool {
        false
    }
    #[inline]
    fn into_owned(self) -> Tensor<T> {
        self
    }
    #[inline]
    fn into_vec(self) -> Vec<T> {
        self.data
    }
}

impl<T> DerefMut for Tensor<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<T: DType> TensorBaseMut<T> for Tensor<T> {}

impl<'a, T> IntoIterator for &'a Tensor<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut Tensor<T> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}

impl<T> IntoIterator for Tensor<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl_tensor_debug!(Tensor);

macro_rules! tensor {
    ($([$([$($x:expr),* $(,)*]),+ $(,)*]),+ $(,)*) => {

    };
    ($([$($x:expr),* $(,)*]),+ $(,)*) => {

    };
    ($($x:expr),* $(,)*) => {
        
    };
}