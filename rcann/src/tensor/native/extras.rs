use crate::tensor::{Dim1, Dim2, Dim3, Dims, ITensor, Tensor, TensorCow, TensorView, TensorViewMut};

macro_rules! impl_tensor_extras {
    ($type_name: ident $(, $l: lifetime )?) => {
        impl<$($l,)?T, D: Dims> std::ops::Index<usize> for $type_name<$($l,)?T, D> {
            type Output = T;
            #[inline]
            fn index(&self, index: usize) -> &Self::Output {
                self.as_ref().index(index)
            }
        }

        impl<$($l,)?T> std::ops::Index<[usize; 2]> for $type_name<$($l,)?T, Dim2> {
            type Output = T;
            #[inline]
            fn index(&self, index: [usize; 2]) -> &Self::Output {
                self.as_ref().index(self.dims().get_compact_offset(&index))
            }
        }

        impl<$($l,)?T> std::ops::Index<[usize; 3]> for $type_name<$($l,)?T, Dim3> {
            type Output = T;
            #[inline]
            fn index(&self, index: [usize; 3]) -> &Self::Output {
                self.as_ref().index(self.dims().get_compact_offset(&index))
            }
        }

        impl<$($l,)?T, D: Dims> PartialEq<Tensor<T, D>> for $type_name<$($l,)?T, D> where T: PartialEq {
            fn eq(&self, other: &Tensor<T, D>) -> bool {
                self.dims() == other.dims() && self.as_ref() == other.as_ref()
            }
        }

        impl<$($l,)?'b, T, D: Dims> PartialEq<TensorView<'b, T, D>> for $type_name<$($l,)?T, D> where T: PartialEq {
            fn eq(&self, other: &TensorView<'b, T, D>) -> bool {
                self.dims() == other.dims() && self.as_ref() == other.as_ref()
            }
        }

        impl<$($l,)?'b, T, D: Dims> PartialEq<TensorViewMut<'b, T, D>> for $type_name<$($l,)?T, D> where T: PartialEq {
            fn eq(&self, other: &TensorViewMut<'b, T, D>) -> bool {
                self.dims() == other.dims() && self.as_ref() == other.as_ref()
            }
        }

        impl<$($l,)?'b, T, D: Dims> PartialEq<TensorCow<'b, T, D>> for $type_name<$($l,)?T, D> where T: PartialEq {
            fn eq(&self, other: &TensorCow<'b, T, D>) -> bool {
                self.dims() == other.dims() && self.as_ref() == other.as_ref()
            }
        }

        impl<$($l,)?T> $type_name<$($l,)?T, Dim1> {
            pub fn as_row_matrix(&self) -> TensorView<T, Dim2> {
                unsafe { TensorView::from_slice_unchecked(self.as_ref(), Dim2(1, self.len()) )}
            }
            pub fn as_col_matrix(&self) -> TensorView<T, Dim2> {
                unsafe { TensorView::from_slice_unchecked(self.as_ref(), Dim2(self.len(), 1) )}
            }
        }

    };
}

macro_rules! impl_tensor_extras_mut {
    ($type_name: ident $(, $l: lifetime )?) => {
        impl<$($l,)?T, D: Dims> std::ops::IndexMut<usize> for $type_name<$($l,)?T, D> {
            #[inline]
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                self.as_mut().index_mut(index)
            }
        }

        impl<$($l,)?T> std::ops::IndexMut<[usize; 2]> for $type_name<$($l,)?T, Dim2> {
            #[inline]
            fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
                let offset = self.dims().get_compact_offset(&index);
                self.as_mut().index_mut(offset)
            }
        }

        impl<$($l,)?T> std::ops::IndexMut<[usize; 3]> for $type_name<$($l,)?T, Dim3> {
            #[inline]
            fn index_mut(&mut self, index: [usize; 3]) -> &mut Self::Output {
                let offset = self.dims().get_compact_offset(&index);
                self.as_mut().index_mut(offset)
            }
        }

        impl<$($l,)?T> $type_name<$($l,)?T, Dim1> {
            pub fn as_row_matrix_mut(&mut self) -> TensorViewMut<T, Dim2> {
                let dims = Dim2(1, self.len());
                unsafe { TensorViewMut::from_slice_unchecked(self.as_mut(),  dims) }
            }
            pub fn as_col_matrix_mut(&mut self) -> TensorViewMut<T, Dim2> {
                let dims = Dim2(self.len(), 1);
                unsafe { TensorViewMut::from_slice_unchecked(self.as_mut(), dims) }
            }
        }
    };
}

impl_tensor_extras!(Tensor);
impl_tensor_extras_mut!(Tensor);
impl_tensor_extras!(TensorView, 'a);
impl_tensor_extras!(TensorViewMut, 'a);
impl_tensor_extras_mut!(TensorViewMut, 'a);
impl_tensor_extras!(TensorCow, 'a);
