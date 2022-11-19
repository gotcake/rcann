use super::base::{TensorBase, TensorBaseMut};
use super::owned::Tensor;
use crate::tensor::{Dims, ITensor};
use std::ops::{Deref, DerefMut};
use std::slice::{Iter, IterMut};

pub struct TensorView<'a, T: 'a> {
    data: &'a [T],
    dims: Dims,
}

impl<'a, T: 'a> TensorView<'a, T> {
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

impl<'a, T: 'a> ITensor<T> for TensorView<'a, T> {
    #[inline]
    fn len(&self) -> usize {
        self.data.len()
    }
    #[inline]
    fn dims(&self) -> &Dims {
        &self.dims
    }
}

impl<'a, T: 'a> Deref for TensorView<'a, T> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<'a, T: 'a> AsRef<[T]> for TensorView<'a, T> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        &self.data
    }
}

impl<'a, T: 'a> TensorBase<T> for TensorView<'a, T> {
    #[inline]
    fn is_owned(&self) -> bool {
        true
    }
    fn into_owned(self) -> Tensor<T> where T: Clone {
        unsafe { Tensor::from_vec_unchecked(self.data.to_vec(), self.dims) }
    }
    fn into_vec(self) -> Vec<T> where T: Clone {
        self.data.to_vec()
    }
}

impl<'a, T: 'a> IntoIterator for &'a TensorView<'a, T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<'a, T: 'a> IntoIterator for TensorView<'a, T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

pub struct TensorViewMut<'a, T: 'a> {
    data: &'a mut [T],
    dims: Dims,
}

impl<'a, T: 'a> ITensor<T> for TensorViewMut<'a, T> {
    #[inline]
    fn len(&self) -> usize {
        self.data.len()
    }
    #[inline]
    fn dims(&self) -> &Dims {
        &self.dims
    }
}

impl<'a, T: 'a> TensorViewMut<'a, T> {
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

impl<'a, T: 'a> Deref for TensorViewMut<'a, T> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<'a, T: 'a> AsRef<[T]> for TensorViewMut<'a, T> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        &self.data
    }
}

impl<'a, T: 'a> TensorBase<T> for TensorViewMut<'a, T> {
    #[inline]
    fn is_owned(&self) -> bool {
        true
    }
    fn into_owned(self) -> Tensor<T> where T: Clone {
        unsafe { Tensor::from_vec_unchecked(self.data.to_vec(), self.dims) }
    }
    fn into_vec(self) -> Vec<T> where T: Clone {
        self.data.to_vec()
    }
}

impl<'a, T: 'a> DerefMut for TensorViewMut<'a, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<'a, T: 'a> AsMut<[T]> for TensorViewMut<'a, T> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
}

impl<'a, T: 'a> TensorBaseMut<T> for TensorViewMut<'a, T> {}

impl<'a, T: 'a> IntoIterator for &'a TensorViewMut<'a, T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<'a, T: 'a> IntoIterator for &'a mut TensorViewMut<'a, T> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}

impl<'a, T: 'a> IntoIterator for TensorViewMut<'a, T> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}
