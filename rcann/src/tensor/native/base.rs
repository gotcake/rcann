use std::ops::{Deref, DerefMut};
use crate::dtype::DType;
use crate::tensor::{ITensorBase, TensorView, TensorViewMut, Tensor, TensorIter, TensorIterMut, Dims};

pub trait TensorBase<T: DType>: ITensorBase<T> + Deref<Target=[T]> {
    fn is_owned(&self) -> bool;
    fn into_owned(self) -> Tensor<T>;
    fn into_vec(self) -> Vec<T>;

    #[inline]
    fn view(&self) -> TensorView<T> {
        unsafe { TensorView::from_slice_unchecked(self, self.dims().clone()) }
    }

    #[inline]
    fn iter_first_axis(&self) -> TensorIter<T> {
        unsafe { TensorIter::new_unchecked(self, self.dims().without_first_axis()) }
    }

    #[inline]
    fn as_1d(&self) -> TensorView<T> {
        unsafe { TensorView::from_slice_unchecked(self, Dims::D1(self.len())) }
    }

    #[inline]
    fn as_row_matrix_2d(&self) -> TensorView<T> {
        unsafe { TensorView::from_slice_unchecked(self, Dims::D2(1, self.len())) }
    }

    #[inline]
    fn as_col_matrix_2d(&self) -> TensorView<T> {
        unsafe { TensorView::from_slice_unchecked(self, Dims::D2(self.len(), 1)) }
    }

    fn contains_non_finite(&self) -> bool {
        !self.iter().all(|x| x.is_finite())
    }

    /*fn iter_first_axis_chunks(&self, len: usize) -> TensorChunkIter<T> {
        let chunk_dims = self.dims().with_resized_first_axis(len);
        unsafe { TensorChunkIter::new_unchecked(self, chunk_dims) }
    }*/
}

pub trait TensorBaseMut<T: DType>: TensorBase<T> + DerefMut<Target=[T]> {
    #[inline]
    fn view_mut(&mut self) -> TensorViewMut<T> {
        let dims = self.dims().clone();
        unsafe { TensorViewMut::from_slice_unchecked(self, dims) }
    }
    #[inline]
    fn iter_first_axis_mut(&mut self) -> TensorIterMut<T> {
        let out_dims = self.dims().without_first_axis();
        unsafe { TensorIterMut::new_unchecked(self, out_dims) }
    }

    #[inline]
    fn as_1d_mut(&mut self) -> TensorViewMut<T> {
        let dims = Dims::D1(self.len());
        unsafe { TensorViewMut::from_slice_unchecked(self, dims) }
    }

    #[inline]
    fn as_row_matrix_2d_mut(&mut self) -> TensorViewMut<T> {
        let dims = Dims::D2(1, self.len());
        unsafe { TensorViewMut::from_slice_unchecked(self, dims) }
    }

    #[inline]
    fn as_col_matrix_2d_mut(&mut self) -> TensorViewMut<T> {
        let dims = Dims::D2(self.len(), 1);
        unsafe { TensorViewMut::from_slice_unchecked(self, dims) }
    }

    /*fn iter_first_axis_chunks_mut(&mut self, len: usize) -> TensorChunkIterMut<T> {
        let chunk_dims = self.dims().with_resized_first_axis(len);
        unsafe { TensorChunkIterMut::new_unchecked(self, chunk_dims) }
    }*/
}

#[macro_export]
macro_rules! impl_tensor_debug {
    ($t: ident $(, $l: lifetime )?) => {
        impl<$($l,)?T: crate::dtype::DType> std::fmt::Debug for $t<$($l,)?T> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                use std::fmt::{Write, Display};
                Display::fmt(&self.dims, f)?;
                f.write_char('[')?;
                if self.len() > 20 {
                    for x in &self[..10] {
                        write!(f, "{:e}, ", x)?;
                    }
                    f.write_str("...")?;
                    for x in &self[self.len()-10..] {
                        write!(f, ", {:e}", x)?;
                    }
                } else {
                    let mut first = true;
                    for x in self {
                        if first {
                            first = false;
                        } else {
                            f.write_str(", ")?;
                        }
                        write!(f, "{:e}", x)?;
                    }
                }
                f.write_char(']')
            }
        }
    };
}
