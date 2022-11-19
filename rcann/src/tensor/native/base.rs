use crate::tensor::{Dims, ITensor, Tensor, TensorIter, TensorIterMut, TensorView, TensorViewMut};

pub trait TensorBase<T>: ITensor<T> + AsRef<[T]> {
    fn is_owned(&self) -> bool;
    fn into_owned(self) -> Tensor<T> where T: Clone;
    fn into_vec(self) -> Vec<T> where T: Clone;

    #[inline]
    fn view(&self) -> TensorView<T> {
        unsafe { TensorView::from_slice_unchecked(self.as_ref(), self.dims().clone()) }
    }

    #[inline]
    fn iter_first_axis(&self) -> TensorIter<T> {
        unsafe { TensorIter::new_unchecked(self.as_ref(), self.dims().without_first_axis()) }
    }

    #[inline]
    fn as_1d(&self) -> TensorView<T> {
        unsafe { TensorView::from_slice_unchecked(self.as_ref(), Dims::D1(self.len())) }
    }

    #[inline]
    fn as_row_matrix_2d(&self) -> TensorView<T> {
        unsafe { TensorView::from_slice_unchecked(self.as_ref(), Dims::D2(1, self.len())) }
    }

    #[inline]
    fn as_col_matrix_2d(&self) -> TensorView<T> {
        unsafe { TensorView::from_slice_unchecked(self.as_ref(), Dims::D2(self.len(), 1)) }
    }

}

pub trait TensorBaseMut<T>: TensorBase<T> + AsMut<[T]> {
    #[inline]
    fn view_mut(&mut self) -> TensorViewMut<T> {
        let dims = self.dims().clone();
        unsafe { TensorViewMut::from_slice_unchecked(self.as_mut(), dims) }
    }
    #[inline]
    fn iter_first_axis_mut(&mut self) -> TensorIterMut<T> {
        let out_dims = self.dims().without_first_axis();
        unsafe { TensorIterMut::new_unchecked(self.as_mut(), out_dims) }
    }

    #[inline]
    fn as_1d_mut(&mut self) -> TensorViewMut<T> {
        let dims = Dims::D1(self.len());
        unsafe { TensorViewMut::from_slice_unchecked(self.as_mut(), dims) }
    }

    #[inline]
    fn as_row_matrix_2d_mut(&mut self) -> TensorViewMut<T> {
        let dims = Dims::D2(1, self.len());
        unsafe { TensorViewMut::from_slice_unchecked(self.as_mut(), dims) }
    }

    #[inline]
    fn as_col_matrix_2d_mut(&mut self) -> TensorViewMut<T> {
        let dims = Dims::D2(self.len(), 1);
        unsafe { TensorViewMut::from_slice_unchecked(self.as_mut(), dims) }
    }

}
