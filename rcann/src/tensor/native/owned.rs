use crate::tensor::dims::{Dim0, Dim1, Dim2, Dim3, Dims};
use crate::tensor::{DimsZero, ITensor, TensorBase, TensorBaseMut};
use num_traits::Zero;
use rand::distributions::Distribution;
use rand::Rng;
use std::mem::ManuallyDrop;
use std::slice::{Iter, IterMut};
use std::vec::IntoIter;

pub struct Tensor<T, D>
where
    D: Dims,
{
    data: Vec<T>,
    dims: D,
}

pub type Tensor0<T> = Tensor<T, Dim0>;
pub type Tensor1<T> = Tensor<T, Dim1>;
pub type Tensor2<T> = Tensor<T, Dim2>;
pub type Tensor3<T> = Tensor<T, Dim3>;

impl<T, D: Dims> Tensor<T, D> {
    pub fn empty() -> Self
    where
        D: DimsZero,
    {
        Tensor {
            data: Vec::new(),
            dims: D::ZERO,
        }
    }

    pub fn from_vec(data: Vec<T>, dims: D) -> Self {
        assert_eq!(data.len(), dims.tensor_len());
        Tensor { data, dims }
    }

    pub fn from_distribution<R, S>(rng: &mut R, dist: S, dims: D) -> Self
    where
        R: Rng,
        S: Distribution<T>,
    {
        let data: Vec<T> = dist.sample_iter(rng).take(dims.tensor_len()).collect();
        Tensor { data, dims }
    }

    #[inline]
    pub(super) unsafe fn from_vec_unchecked(data: Vec<T>, dim: D) -> Self {
        debug_assert_eq!(data.len(), dim.tensor_len());
        Tensor { data, dims: dim }
    }
}

impl<T> Tensor0<T> {
    pub fn scalar(value: T) -> Self {
        Tensor {
            data: vec![value],
            dims: Dim0,
        }
    }
}

impl<T> Tensor1<T> {
    pub fn from_vec_1d(data: Vec<T>) -> Self {
        let len = data.len();
        Tensor { data, dims: Dim1(len) }
    }
}

impl<T> Tensor2<T> {
    pub fn from_vec_2d<const N: usize>(vec: Vec<[T; N]>) -> Self {
        unsafe {
            let mut vec = ManuallyDrop::new(vec);
            let (ptr, len, cap) = (vec.as_mut_ptr(), vec.len(), vec.capacity());
            let data = Vec::from_raw_parts(ptr as *mut T, len * N, cap * N);
            Tensor::from_vec_unchecked(data, Dim2(vec.len(), N))
        }
    }
}

impl<T> Tensor3<T> {
    pub fn from_vec_3d<const N: usize, const M: usize>(vec: Vec<[[T; M]; N]>) -> Self {
        let inner_size: usize = N * M;
        unsafe {
            let mut vec = ManuallyDrop::new(vec);
            let (ptr, len, cap) = (vec.as_mut_ptr(), vec.len(), vec.capacity());
            let data = Vec::from_raw_parts(ptr as *mut T, len * inner_size, cap * inner_size);
            Tensor::from_vec_unchecked(data, Dim3(vec.len(), N, M))
        }
    }
}

impl<T: Clone, D: Dims> Tensor<T, D> {
    pub fn filled(value: T, dims: D) -> Self {
        Tensor {
            data: vec![value; dims.tensor_len()],
            dims,
        }
    }
    pub fn resize(&mut self, fill: T, dims: D) {
        if self.dims != dims {
            let len = self.data.len();
            let new_len = dims.tensor_len();
            if len != new_len {
                self.data.resize(new_len, fill);
            }
            self.dims = dims;
        }
    }
    pub fn resize_within_capacity(&mut self, fill: T, dims: D) {
        if self.dims != dims {
            let new_len = dims.tensor_len();
            if new_len > self.data.capacity() {
                panic!(
                    "Dims {dims} with length {new_len} not within capacity {}",
                    self.data.capacity()
                );
            }
            let len = self.data.len();
            if len != new_len {
                self.data.resize(new_len, fill);
            }
            self.dims = dims;
        }
    }
    #[inline]
    pub fn fill(&mut self, fill: T) {
        self.data.fill(fill);
    }
}

impl<T: Zero + Clone, D: Dims> Tensor<T, D> {
    #[inline]
    pub fn zeroed(dims: D) -> Self {
        Self::filled(T::zero(), dims)
    }
    #[inline]
    pub fn fill_zero(&mut self) {
        self.data.fill(T::zero());
    }
}

impl<T, D: Dims> ITensor<D> for Tensor<T, D> {
    #[inline]
    fn len(&self) -> usize {
        self.data.len()
    }
    #[inline]
    fn dims(&self) -> &D {
        &self.dims
    }
}

impl<T, D: Dims> AsRef<[T]> for Tensor<T, D> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        &self.data
    }
}

impl<T, D: Dims> TensorBase<T, D> for Tensor<T, D> {
    #[inline]
    fn is_owned(&self) -> bool {
        false
    }
    #[inline]
    fn into_owned(self) -> Tensor<T, D> {
        self
    }
    #[inline]
    fn into_vec(self) -> Vec<T> {
        self.data
    }
}

impl<T, D: Dims> AsMut<[T]> for Tensor<T, D> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
}

impl<T, D: Dims> TensorBaseMut<T, D> for Tensor<T, D> {}

impl<'a, T, D: Dims> IntoIterator for &'a Tensor<T, D> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<'a, T, D: Dims> IntoIterator for &'a mut Tensor<T, D> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}

impl<T, D: Dims> IntoIterator for Tensor<T, D> {
    type Item = T;
    type IntoIter = IntoIter<T>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<T: Clone, D: Dims> Clone for Tensor<T, D> {
    fn clone(&self) -> Self {
        unsafe { Tensor::from_vec_unchecked(self.data.clone(), self.dims) }
    }
}

#[macro_export]
macro_rules! tensor {
    ($([$([$($x:expr),* $(,)*]),+ $(,)*]),+ $(,)*) => {
        $crate::tensor::Tensor3::from_vec_3d(vec![$([$([$($x,)*],)*],)*])
    };
    ($([$($x:expr),* $(,)*]),+ $(,)*) => {
        $crate::tensor::Tensor2::from_vec_2d(vec![$([$($x,)*],)*])
    };
    ($($x:expr),* $(,)*) => {
        $crate::tensor::Tensor1::from_vec_1d(vec![$($x,)*])
    };
}
