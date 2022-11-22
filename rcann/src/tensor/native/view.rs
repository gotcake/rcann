use super::base::{TensorBase, TensorBaseMut};
use super::owned::Tensor;
use crate::tensor::dims::{Dim0, Dim1, Dim2, Dim3, Dims};
use crate::tensor::ITensor;
use std::slice::{Iter, IterMut};

pub struct TensorView<'a, T: 'a, D: Dims> {
    data: &'a [T],
    dims: D,
}

pub type TensorView0<'a, T> = TensorView<'a, T, Dim0>;
pub type TensorView1<'a, T> = TensorView<'a, T, Dim1>;
pub type TensorView2<'a, T> = TensorView<'a, T, Dim2>;
pub type TensorView3<'a, T> = TensorView<'a, T, Dim3>;

impl<'a, T: 'a, D: Dims> TensorView<'a, T, D> {
    pub fn from_slice(data: &'a [T], dims: D) -> Self {
        assert_eq!(
            data.len(),
            dims.tensor_len(),
            "Mismatched data length {} and dimension {:?}",
            data.len(),
            dims
        );
        TensorView { data, dims }
    }
    #[inline]
    pub(super) unsafe fn from_slice_unchecked(data: &'a [T], dims: D) -> Self {
        debug_assert_eq!(data.len(), dims.tensor_len());
        TensorView { data, dims }
    }
}

impl<'a, T: 'a, D: Dims> ITensor<D> for TensorView<'a, T, D> {
    #[inline]
    fn len(&self) -> usize {
        self.data.len()
    }
    #[inline]
    fn dims(&self) -> &D {
        &self.dims
    }
}

impl<'a, T: 'a, D: Dims> AsRef<[T]> for TensorView<'a, T, D> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        &self.data
    }
}

impl<'a, T: 'a, D: Dims> TensorBase<T, D> for TensorView<'a, T, D> {
    #[inline]
    fn is_owned(&self) -> bool {
        true
    }
    fn into_owned(self) -> Tensor<T, D>
    where
        T: Clone,
    {
        unsafe { Tensor::from_vec_unchecked(self.data.to_vec(), self.dims) }
    }
    fn into_vec(self) -> Vec<T>
    where
        T: Clone,
    {
        self.data.to_vec()
    }
}

impl<'a, T: 'a, D: Dims> IntoIterator for &'a TensorView<'a, T, D> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<'a, T: 'a, D: Dims> IntoIterator for TensorView<'a, T, D> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

pub struct TensorViewMut<'a, T: 'a, D: Dims> {
    data: &'a mut [T],
    dims: D,
}

pub type TensorViewMut0<'a, T> = TensorViewMut<'a, T, Dim0>;
pub type TensorViewMut1<'a, T> = TensorViewMut<'a, T, Dim1>;
pub type TensorViewMut2<'a, T> = TensorViewMut<'a, T, Dim2>;
pub type TensorViewMut3<'a, T> = TensorViewMut<'a, T, Dim3>;

impl<'a, T: 'a, D: Dims> ITensor<D> for TensorViewMut<'a, T, D> {
    #[inline]
    fn len(&self) -> usize {
        self.data.len()
    }
    #[inline]
    fn dims(&self) -> &D {
        &self.dims
    }
}

impl<'a, T: 'a, D: Dims> TensorViewMut<'a, T, D> {
    pub fn from_slice(data: &'a mut [T], dim: D) -> Self {
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
    pub(super) unsafe fn from_slice_unchecked(data: &'a mut [T], dim: D) -> Self {
        debug_assert_eq!(data.len(), dim.tensor_len());
        TensorViewMut { data, dims: dim }
    }
}

impl<'a, T: 'a, D: Dims> AsRef<[T]> for TensorViewMut<'a, T, D> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        &self.data
    }
}

impl<'a, T: 'a, D: Dims> TensorBase<T, D> for TensorViewMut<'a, T, D> {
    #[inline]
    fn is_owned(&self) -> bool {
        true
    }
    fn into_owned(self) -> Tensor<T, D>
    where
        T: Clone,
    {
        unsafe { Tensor::from_vec_unchecked(self.data.to_vec(), self.dims) }
    }
    fn into_vec(self) -> Vec<T>
    where
        T: Clone,
    {
        self.data.to_vec()
    }
}

impl<'a, T: 'a, D: Dims> AsMut<[T]> for TensorViewMut<'a, T, D> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
}

impl<'a, T: 'a, D: Dims> TensorBaseMut<T, D> for TensorViewMut<'a, T, D> {}

impl<'a, T: 'a, D: Dims> IntoIterator for &'a TensorViewMut<'a, T, D> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<'a, T: 'a, D: Dims> IntoIterator for &'a mut TensorViewMut<'a, T, D> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}

impl<'a, T: 'a, D: Dims> IntoIterator for TensorViewMut<'a, T, D> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}
