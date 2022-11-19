use std::ops::Deref;
use std::slice::Iter;
use crate::tensor::{Dims, ITensor, Tensor, TensorBase};

enum Holder<'a, T> {
    Borrowed(&'a [T]),
    Owned(Vec<T>)
}

impl<'a, T> Holder<'a, T> {
    fn is_owned(&self) -> bool {
        use Holder::*;
        match self {
            Borrowed(_) => false,
            Owned(_) => true
        }
    }
    fn into_owned(self) -> Vec<T> where T: Clone {
        use Holder::*;
        match self {
            Borrowed(data) => data.to_vec(),
            Owned(data) => data,
        }
    }
}

impl<'a, T> Deref for Holder<'a, T> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &Self::Target {
        use Holder::*;
        match self {
            &Borrowed(data) => data,
            Owned(data) => data.as_slice(),
        }
    }
}

pub struct TensorCow<'a, T> {
    data: Holder<'a, T>,
    dims: Dims,
}

impl<'a, T> TensorCow<'a, T> {
    pub fn borrowed<D>(data: &'a [T], dims: D) -> Self where D: Into<Dims> {
        let dims = dims.into();
        assert_eq!(
            data.len(),
            dims.tensor_len(),
            "Mismatched data length {} and dimension {:?}",
            data.len(),
            dims
        );
        TensorCow { data: Holder::Borrowed(data), dims }
    }

    #[inline]
    pub(super) unsafe fn borrowed_unchecked(data: &'a [T], dims: Dims) -> Self {
        debug_assert_eq!(data.len(), dims.tensor_len());
        TensorCow { data: Holder::Borrowed(data), dims }
    }
}

impl<T> TensorCow<'static, T> {
    pub fn owned<D>(data: Vec<T>, dims: D) -> Self where D: Into<Dims> {
        let dims = dims.into();
        assert_eq!(
            data.len(),
            dims.tensor_len(),
            "Mismatched data length {} and dimension {:?}",
            data.len(),
            dims
        );
        TensorCow { data: Holder::Owned(data), dims }
    }
    #[inline]
    pub(super) unsafe fn owned_unchecked(data: Vec<T>, dims: Dims) -> Self {
        debug_assert_eq!(data.len(), dims.tensor_len());
        TensorCow { data: Holder::Owned(data), dims }
    }
}

impl<'a, T> ITensor<T> for TensorCow<'a, T> {
    #[inline]
    fn len(&self) -> usize {
        self.data.len()
    }
    #[inline]
    fn dims(&self) -> &Dims {
        &self.dims
    }
}

impl<'a, T> Deref for TensorCow<'a, T> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<'a, T> AsRef<[T]> for TensorCow<'a, T> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        &self.data
    }
}

impl<'a, T> TensorBase<T> for TensorCow<'a, T> {

    #[inline]
    fn is_owned(&self) -> bool {
        self.data.is_owned()
    }

    fn into_owned(self) -> Tensor<T> where T: Clone {
        unsafe { Tensor::from_vec_unchecked(self.data.into_owned(), self.dims) }
    }

    #[inline]
    fn into_vec(self) -> Vec<T> where T: Clone {
        self.data.into_owned()
    }
}

impl<'a, T> IntoIterator for &'a TensorCow<'a, T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}


