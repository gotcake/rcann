use super::base::{TensorBase, TensorBaseMut};
use super::tensor::Tensor;
use crate::dtype::DType;
use crate::impl_tensor_debug;
use crate::tensor::{Dims, ITensorBase};
use std::ops::{Deref, DerefMut};
use std::slice::{Iter, IterMut};

pub struct TensorView<'a, T> {
    data: &'a [T],
    dims: Dims,
}

impl<'a, T> TensorView<'a, T> {
    pub fn from_slice<D: Into<Dims>>(data: &'a [T], dim: D) -> Self {
        let dim = dim.into();
        assert_eq!(
            data.len(),
            dim.tensor_len(),
            "Mismatched data length {} and dimension {:?}",
            data.len(),
            dim
        );
        TensorView { data, dims: dim }
    }
    #[inline]
    pub(super) unsafe fn from_slice_unchecked(data: &'a [T], dim: Dims) -> Self {
        debug_assert_eq!(data.len(), dim.tensor_len());
        TensorView { data, dims: dim }
    }
}

impl<'a, T: DType> ITensorBase<T> for TensorView<'a, T> {
    #[inline]
    fn len(&self) -> usize {
        self.data.len()
    }
    #[inline]
    fn dims(&self) -> &Dims {
        &self.dims
    }
}

impl<'a, T> Deref for TensorView<'a, T> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<'a, T: DType> TensorBase<T> for TensorView<'a, T> {
    #[inline]
    fn is_owned(&self) -> bool {
        true
    }
    fn into_owned(self) -> Tensor<T> {
        unsafe { Tensor::from_vec_unchecked(self.data.to_vec(), self.dims) }
    }
    fn into_vec(self) -> Vec<T> {
        self.data.to_vec()
    }
}

impl<'a, T> IntoIterator for &'a TensorView<'a, T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<'a, T> IntoIterator for TensorView<'a, T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl_tensor_debug!(TensorView, 'a);

pub struct TensorViewMut<'a, T> {
    data: &'a mut [T],
    dims: Dims,
}

impl<'a, T: DType> ITensorBase<T> for TensorViewMut<'a, T> {
    #[inline]
    fn len(&self) -> usize {
        self.data.len()
    }
    #[inline]
    fn dims(&self) -> &Dims {
        &self.dims
    }
}

impl<'a, T> TensorViewMut<'a, T> {
    pub fn from_slice<D: Into<Dims>>(data: &'a mut [T], dim: D) -> Self {
        let dim = dim.into();
        assert_eq!(
            data.len(),
            dim.tensor_len(),
            "Mismatched data length {} and dimension {:?}",
            data.len(),
            dim
        );
        TensorViewMut { data, dims: dim }
    }
    #[inline]
    pub(super) unsafe fn from_slice_unchecked(data: &'a mut [T], dim: Dims) -> Self {
        debug_assert_eq!(data.len(), dim.tensor_len());
        TensorViewMut { data, dims: dim }
    }
}

impl<'a, T> Deref for TensorViewMut<'a, T> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<'a, T: DType> TensorBase<T> for TensorViewMut<'a, T> {
    #[inline]
    fn is_owned(&self) -> bool {
        true
    }
    fn into_owned(self) -> Tensor<T> {
        unsafe { Tensor::from_vec_unchecked(self.data.to_vec(), self.dims) }
    }
    fn into_vec(self) -> Vec<T> {
        self.data.to_vec()
    }
}

impl<'a, T> DerefMut for TensorViewMut<'a, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<'a, T: DType> TensorBaseMut<T> for TensorViewMut<'a, T> {}

impl<'a, T> IntoIterator for &'a TensorViewMut<'a, T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut TensorViewMut<'a, T> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}

impl<'a, T> IntoIterator for TensorViewMut<'a, T> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}

impl_tensor_debug!(TensorViewMut, 'a);
