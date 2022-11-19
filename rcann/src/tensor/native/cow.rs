use crate::tensor::dims::{Dim0, Dim1, Dim2, Dim3, Dims};
use crate::tensor::{ITensor, Tensor, TensorBase};
use std::ops::Deref;
use std::slice::Iter;

enum Holder<'a, T: 'a> {
    Borrowed(&'a [T]),
    Owned(Vec<T>),
}

impl<'a, T> Holder<'a, T> {
    fn is_owned(&self) -> bool {
        use Holder::*;
        match self {
            Borrowed(_) => false,
            Owned(_) => true,
        }
    }
    fn into_owned(self) -> Vec<T>
    where
        T: Clone,
    {
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

pub struct TensorCow<'a, T: 'a, D: Dims> {
    data: Holder<'a, T>,
    dims: D,
}

pub type TensorCow0<'a, T> = TensorCow<'a, T, Dim0>;
pub type TensorCow1<'a, T> = TensorCow<'a, T, Dim1>;
pub type TensorCow2<'a, T> = TensorCow<'a, T, Dim2>;
pub type TensorCow3<'a, T> = TensorCow<'a, T, Dim3>;

impl<'a, T: 'a, D: Dims> TensorCow<'a, T, D> {
    pub fn borrowed(data: &'a [T], dims: D) -> Self {
        assert_eq!(
            data.len(),
            dims.tensor_len(),
            "Mismatched data length {} and dimension {:?}",
            data.len(),
            dims
        );
        TensorCow {
            data: Holder::Borrowed(data),
            dims,
        }
    }

    #[inline]
    pub(super) unsafe fn borrowed_unchecked(data: &'a [T], dims: D) -> Self {
        debug_assert_eq!(data.len(), dims.tensor_len());
        TensorCow {
            data: Holder::Borrowed(data),
            dims,
        }
    }
}

impl<T, D: Dims> TensorCow<'static, T, D> {
    pub fn owned(data: Vec<T>, dims: D) -> Self {
        assert_eq!(
            data.len(),
            dims.tensor_len(),
            "Mismatched data length {} and dimension {:?}",
            data.len(),
            dims
        );
        TensorCow {
            data: Holder::Owned(data),
            dims,
        }
    }
    #[inline]
    pub(super) unsafe fn owned_unchecked(data: Vec<T>, dims: D) -> Self {
        debug_assert_eq!(data.len(), dims.tensor_len());
        TensorCow {
            data: Holder::Owned(data),
            dims,
        }
    }
}

impl<'a, T: 'a, D: Dims> ITensor<T, D> for TensorCow<'a, T, D> {
    #[inline]
    fn len(&self) -> usize {
        self.data.len()
    }
    #[inline]
    fn dims(&self) -> &D {
        &self.dims
    }
}

impl<'a, T: 'a, D: Dims> AsRef<[T]> for TensorCow<'a, T, D> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        &self.data
    }
}

impl<'a, T: 'a, D: Dims> TensorBase<T, D> for TensorCow<'a, T, D> {
    #[inline]
    fn is_owned(&self) -> bool {
        self.data.is_owned()
    }

    fn into_owned(self) -> Tensor<T, D>
    where
        T: Clone,
    {
        unsafe { Tensor::from_vec_unchecked(self.data.into_owned(), self.dims) }
    }

    #[inline]
    fn into_vec(self) -> Vec<T>
    where
        T: Clone,
    {
        self.data.into_owned()
    }
}

impl<'a, T: 'a, D: Dims> IntoIterator for &'a TensorCow<'a, T, D> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}
