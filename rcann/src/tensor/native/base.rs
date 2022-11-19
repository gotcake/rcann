use crate::tensor::{
    Dim1, Dims, ITensor, Tensor, TensorIter, TensorIterMut, TensorView, TensorView1, TensorViewMut,
    TensorViewMut1,
};
use std::slice::{Iter, IterMut};

pub trait TensorBase<T, D: Dims>: ITensor<T, D> + AsRef<[T]> {
    fn is_owned(&self) -> bool;
    fn into_owned(self) -> Tensor<T, D>
    where
        T: Clone;
    fn into_vec(self) -> Vec<T>
    where
        T: Clone;

    #[inline]
    fn view(&self) -> TensorView<T, D> {
        unsafe { TensorView::from_slice_unchecked(self.as_ref(), self.dims().clone()) }
    }

    #[inline]
    fn iter_first_axis(&self) -> TensorIter<T, D::Less> {
        unsafe { TensorIter::new_unchecked(self.as_ref(), self.dims().without_first_axis()) }
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

pub trait TensorBaseMut<T, D>: TensorBase<T, D> + AsMut<[T]>
where
    D: Dims,
{
    #[inline]
    fn view_mut(&mut self) -> TensorViewMut<T, D> {
        let dims = self.dims().clone();
        unsafe { TensorViewMut::from_slice_unchecked(self.as_mut(), dims) }
    }
    #[inline]
    fn iter_first_axis_mut(&mut self) -> TensorIterMut<T, D::Less> {
        let out_dims = self.dims().without_first_axis();
        unsafe { TensorIterMut::new_unchecked(self.as_mut(), out_dims) }
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
