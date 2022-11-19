use std::mem::ManuallyDrop;
use crate::tensor::{Dims, ITensor, TensorBase, TensorBaseMut};
use std::ops::{Deref, DerefMut};
use std::slice::{Iter, IterMut};
use std::vec::IntoIter;
use rand::distributions::Distribution;
use rand::Rng;

pub struct Tensor<T> {
    data: Vec<T>,
    dims: Dims,
}

impl<T> Tensor<T> {
    pub fn empty() -> Tensor<T> {
        Tensor {
            data: Vec::new(),
            dims: Dims::D1(0),
        }
    }
    pub fn scalar(value: T) -> Tensor<T> {
        Tensor {
            data: vec![value],
            dims: Dims::D0,
        }
    }
    pub fn from_vec<D: Into<Dims>>(data: Vec<T>, dim: D) -> Tensor<T> {
        let dim = dim.into();
        assert_eq!(
            data.len(),
            dim.tensor_len(),
            "Mismatched data length {} and dimension {:?}",
            data.len(),
            dim
        );
        Tensor { data, dims: dim }
    }

    pub fn from_distribution<R, S, D>(rng: &mut R, dist: S, dims: D) -> Self where R: Rng, S: Distribution<T>, D: Into<Dims> {
        let dims = dims.into();
        let data: Vec<T> = dist
            .sample_iter(rng)
            .take(dims.tensor_len())
            .collect();
        Tensor {
            data,
            dims,
        }
    }

    #[inline]
    pub(super) unsafe fn from_vec_unchecked(data: Vec<T>, dim: Dims) -> Tensor<T> {
        debug_assert_eq!(data.len(), dim.tensor_len());
        Tensor { data, dims: dim }
    }

    pub fn from_vec_1d(vec: Vec<T>) -> Tensor<T> {
        let len = vec.len();
        unsafe { Tensor::from_vec_unchecked(vec, Dims::D1(len)) }
    }

    pub fn from_vec_2d<const N: usize>(vec: Vec<[T; N]>) -> Tensor<T> {
        unsafe {
            let mut vec = ManuallyDrop::new(vec);
            let (ptr, len, cap) = (vec.as_mut_ptr(), vec.len(), vec.capacity());
            let data= Vec::from_raw_parts(ptr as *mut T, len * N, cap * N);
            Tensor::from_vec_unchecked(data, Dims::D2(vec.len(), N))
        }
    }

    pub fn from_vec_3d<const N: usize, const M: usize>(vec: Vec<[[T; M]; N]>) -> Tensor<T> {
        let inner_size: usize = N * M;
        unsafe {
            let mut vec = ManuallyDrop::new(vec);
            let (ptr, len, cap) = (vec.as_mut_ptr(), vec.len(), vec.capacity());
            let data= Vec::from_raw_parts(ptr as *mut T, len * inner_size, cap * inner_size);
            Tensor::from_vec_unchecked(data, Dims::D3(vec.len(), N, M))
        }
    }

}

impl<T: Clone> Tensor<T> {
    pub fn filled<D>(value: T, dims: D) -> Self where D: Into<Dims> {
        let dims = dims.into();
        Tensor {
            data: vec![value; dims.tensor_len()],
            dims
        }
    }
    pub fn resize_fill<D>(&mut self, dims: D, fill: T) where D: Into<Dims> {
        let dims = dims.into();
        if self.dims != dims {
            let len = self.data.len();
            let new_len = dims.tensor_len();
            if len != new_len {
                self.data.resize(new_len, fill);
            }
        }
    }
}


impl<T: Default + Clone> Tensor<T> {
    pub fn filled_default<D>(dims: D) -> Self where D: Into<Dims> {
        Self::filled(T::default(), dims)
    }
    pub fn resize_fill_default<D>(&mut self, dims: D) where D: Into<Dims> {
        self.resize_fill(dims, T::default());
    }
}

impl<T> ITensor<T> for Tensor<T> {
    #[inline]
    fn len(&self) -> usize {
        self.data.len()
    }
    #[inline]
    fn dims(&self) -> &Dims {
        &self.dims
    }
}

impl<T> Deref for Tensor<T> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T> AsRef<[T]> for Tensor<T> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        &self.data
    }
}

impl<T> TensorBase<T> for Tensor<T> {
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

impl<T> AsMut<[T]> for Tensor<T> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
}

impl<T> TensorBaseMut<T> for Tensor<T> {}

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

#[macro_export]
macro_rules! tensor {
    ($([$([$($x:expr),* $(,)*]),+ $(,)*]),+ $(,)*) => {
        $crate::tensor::Tensor::from_vec_3d(vec![$([$([$($x,)*],)*],)*])
    };
    ($([$($x:expr),* $(,)*]),+ $(,)*) => {
        $crate::tensor::Tensor::from_vec_2d(vec![$([$($x,)*],)*])
    };
    ($($x:expr),* $(,)*) => {
        $crate::tensor::Tensor::from_vec_1d(vec![$($x,)*])
    };
}
