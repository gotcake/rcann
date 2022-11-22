use std::ops::{Index, IndexMut};
use crate::tensor::{Dim1, Dims, DimsMore, ITensor, Tensor, TensorChunkIter, TensorChunkIterMut, TensorIter, TensorIterMut, TensorView, TensorView1, TensorViewMut, TensorViewMut1};
use std::slice::{Iter, IterMut};

pub trait TensorBase<T, D: Dims>: ITensor<D> + AsRef<[T]> + Index<usize, Output=T> {
    fn is_owned(&self) -> bool;
    fn into_owned(self) -> Tensor<T, D>
    where
        T: Clone;
    fn into_vec(self) -> Vec<T>
    where
        T: Clone;

    #[inline]
    fn view(&self) -> TensorView<T, D> {
        unsafe { TensorView::from_slice_unchecked(self.as_ref(), *self.dims()) }
    }

    #[inline]
    fn iter_major_axis(&self) -> TensorIter<T, D::Less> {
        unsafe { TensorIter::new_unchecked(self.as_ref(), self.dims().remove_major()) }
    }

    fn iter_major_axis_chunks(&self, size: usize) -> TensorChunkIter<T, D::Less> {
        unsafe { TensorChunkIter::new_unchecked(self.as_ref(), self.dims().remove_major(), size) }
    }

    #[inline]
    fn flatten(&self) -> TensorView1<T> {
        unsafe { TensorView::from_slice_unchecked(self.as_ref(), Dim1(self.len())) }
    }

    #[inline]
    fn iter(&self) -> Iter<T> {
        self.as_ref().iter()
    }
}

pub trait TensorBaseMut<T, D: Dims>: TensorBase<T, D> + AsMut<[T]> + IndexMut<usize, Output=T> {
    #[inline]
    fn view_mut(&mut self) -> TensorViewMut<T, D> {
        let dims = self.dims().clone();
        unsafe { TensorViewMut::from_slice_unchecked(self.as_mut(), dims) }
    }
    #[inline]
    fn iter_major_axis_mut(&mut self) -> TensorIterMut<T, D::Less> {
        let out_dims = self.dims().remove_major();
        unsafe { TensorIterMut::new_unchecked(self.as_mut(), out_dims) }
    }

    fn iter_major_axis_chunks_mut(&mut self, size: usize) -> TensorChunkIterMut<T, D::Less> {
        let inner_dims = self.dims().remove_major();
        unsafe { TensorChunkIterMut::new_unchecked(self.as_mut(), inner_dims, size) }
    }

    #[inline]
    fn flatten_mut(&mut self) -> TensorViewMut1<T> {
        let dims = Dim1(self.len());
        unsafe { TensorViewMut::from_slice_unchecked(self.as_mut(), dims) }
    }

    #[inline]
    fn iter_mut(&mut self) -> IterMut<T> {
        self.as_mut().iter_mut()
    }
}
