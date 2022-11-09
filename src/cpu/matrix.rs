use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::slice::{ChunksExact, ChunksExactMut};
use num_traits::Zero;
use crate::dtype::DType;
use crate::tensor::{Tensor, Tensor2};

pub fn resize_vec<T: Zero + Clone>(vec: &mut Vec<T>, len: usize) {
    let old_len = vec.len();
    if old_len != len {
        if vec.capacity() < len {
            vec.reserve_exact(len);
        }
        unsafe { vec.set_len(len); }
        if old_len < len {
            vec[old_len..len].fill(T::zero());
        }
    }
}

impl<T: DType> Tensor<T> for Vec<T> {
    #[inline]
    fn len(&self) -> usize {
        Vec::len(self)
    }
}

pub trait MatrixBase<T: DType>:
    Tensor2<T> + Deref<Target=[T]> + Index<(usize, usize), Output=T>
{
    fn iter_rows(&self) -> ChunksExact<'_, T>;
}

pub trait MatrixBaseMut<T: DType>:
    MatrixBase<T> + DerefMut + IndexMut<(usize, usize), Output=T>
{
    fn iter_rows_mut(&mut self) -> ChunksExactMut<'_, T>;
}

#[derive(Clone)]
pub struct Matrix<T: DType> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
}

impl<T: DType> Tensor<T> for Matrix<T> {
    #[inline]
    fn len(&self) -> usize {
        self.data.len()
    }
}

impl<T: DType> Tensor2<T> for Matrix<T> {
    #[inline]
    fn rows(&self) -> usize {
        self.rows
    }
    #[inline]
    fn cols(&self) -> usize {
        self.cols
    }
    #[inline]
    fn dim(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
}

impl<T: DType> MatrixBase<T> for Matrix<T> {
    #[inline]
    fn iter_rows(&self) -> ChunksExact<'_, T> {
        self.data.chunks_exact(self.cols)
    }
}

impl <T: DType> MatrixBaseMut<T> for Matrix<T> {
    #[inline]
    fn iter_rows_mut(&mut self) -> ChunksExactMut<'_, T> {
        self.data.chunks_exact_mut(self.cols)
    }
}

impl<T: DType> Matrix<T> {

    pub fn empty() -> Self {
        Matrix {
            rows: 0,
            cols: 0,
            data: Vec::new(),
        }
    }

    pub fn zero(rows: usize, cols: usize) -> Self {
        let len = rows.checked_mul(cols).expect("dimensions too large");
        Matrix {
            rows,
            cols,
            data: vec![T::zero(); len],
        }
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<T>) -> Self {
        assert_eq!(data.len(), rows.checked_mul(cols).expect("invalid dimensions"));
        Matrix {
            rows,
            cols,
            data,
        }
    }

    #[inline]
    pub fn view(&self) -> MatrixRef<T> {
        MatrixRef::from_slice(self.rows, self.cols, &self.data)
    }

    #[inline]
    pub fn view_mut(&mut self) -> MatrixRefMut<T> {
        MatrixRefMut::from_slice(self.rows, self.cols, &mut self.data)
    }

    #[inline]
    pub fn resize(&mut self, rows: usize, cols: usize) {
        let len = rows.checked_mul(cols).expect("dimensions too large");
        resize_vec(&mut self.data, len);
        self.rows = rows;
        self.cols = cols;
    }

}

impl<T: DType> Deref for Matrix<T> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T: DType> DerefMut for Matrix<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<T: DType> Index<(usize, usize)> for Matrix<T> {
    type Output = T;
    #[inline]
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        assert!(index.0 < self.rows && index.1 < self.cols, "Index out of bounds: {:?}, size=({},{})", index, self.rows, self.cols);
        unsafe { self.data.get_unchecked(index.0 * self.cols + index.1) }
    }
}

impl<T: DType> IndexMut<(usize, usize)> for Matrix<T> {
    #[inline]
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        assert!(index.0 < self.rows && index.1 < self.cols, "Index out of bounds: {:?}, size=({},{})", index, self.rows, self.cols);
        unsafe { self.data.get_unchecked_mut(index.0 * self.cols + index.1) }
    }
}

pub struct MatrixRef<'a, T: DType> {
    data: &'a [T],
    rows: usize,
    cols: usize,
}

impl<'a, T: DType> MatrixRef<'a, T> {
    #[inline]
    pub fn from_slice(rows: usize, cols: usize, slice: &'a[T]) -> Self {
        assert_eq!(slice.len(), rows * cols);
        MatrixRef {
            data: slice,
            rows,
            cols
        }
    }
}

impl<'a, T: DType> Deref for MatrixRef<'a, T> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<'a, T: DType> Index<(usize, usize)> for MatrixRef<'a, T> {
    type Output = T;
    #[inline]
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        assert!(index.0 < self.rows && index.1 < self.cols, "Index out of bounds: {:?}, size=({},{})", index, self.rows, self.cols);
        unsafe { self.data.get_unchecked(index.0 * self.cols + index.1) }
    }
}


impl<'a, T: DType> Tensor<T> for MatrixRef<'a, T> {
    #[inline]
    fn len(&self) -> usize {
        self.data.len()
    }
}

impl<'a, T: DType> Tensor2<T> for MatrixRef<'a, T> {
    #[inline]
    fn rows(&self) -> usize {
        self.rows
    }
    #[inline]
    fn cols(&self) -> usize {
        self.cols
    }
    #[inline]
    fn dim(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
}

impl<'a, T: DType> MatrixBase<T> for MatrixRef<'a, T> {
    #[inline]
    fn iter_rows(&self) -> ChunksExact<'_, T> {
        self.data.chunks_exact(self.cols)
    }
}

pub struct MatrixRefMut<'a, T: DType> {
    data: &'a mut [T],
    rows: usize,
    cols: usize,
}

impl<'a, T: DType> MatrixRefMut<'a, T> {
    #[inline]
    pub fn from_slice(rows: usize, cols: usize, slice: &'a mut [T]) -> Self {
        assert_eq!(slice.len(), rows * cols);
        MatrixRefMut {
            data: slice,
            rows,
            cols
        }
    }
}

impl<'a, T: DType> Deref for MatrixRefMut<'a, T> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<'a, T: DType> DerefMut for MatrixRefMut<'a, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<'a, T: DType> Index<(usize, usize)> for MatrixRefMut<'a, T> {
    type Output = T;
    #[inline]
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        assert!(index.0 < self.rows && index.1 < self.cols, "Index out of bounds: {:?}, size=({},{})", index, self.rows, self.cols);
        unsafe { self.data.get_unchecked(index.0 * self.cols + index.1) }
    }
}

impl<'a, T: DType> IndexMut<(usize, usize)> for MatrixRefMut<'a, T> {
    #[inline]
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        assert!(index.0 < self.rows && index.1 < self.cols, "Index out of bounds: {:?}, size=({},{})", index, self.rows, self.cols);
        unsafe { self.data.get_unchecked_mut(index.0 * self.cols + index.1) }
    }
}

impl<'a, T: DType> Tensor<T> for MatrixRefMut<'a, T> {
    #[inline]
    fn len(&self) -> usize {
        self.data.len()
    }
}

impl<'a, T: DType> Tensor2<T> for MatrixRefMut<'a, T> {
    #[inline]
    fn rows(&self) -> usize {
        self.rows
    }
    #[inline]
    fn cols(&self) -> usize {
        self.cols
    }
    #[inline]
    fn dim(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
}

impl<'a, T: DType> MatrixBase<T> for MatrixRefMut<'a, T> {
    #[inline]
    fn iter_rows(&self) -> ChunksExact<'_, T> {
        self.data.chunks_exact(self.cols)
    }
}

impl<'a, T: DType> MatrixBaseMut<T> for MatrixRefMut<'a, T> {
    #[inline]
    fn iter_rows_mut(&mut self) -> ChunksExactMut<'_, T> {
        self.data.chunks_exact_mut(self.cols)
    }
}

