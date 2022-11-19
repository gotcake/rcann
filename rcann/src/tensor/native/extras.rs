use crate::tensor::{ITensor, Tensor, TensorCow, TensorView, TensorViewMut};

macro_rules! impl_tensor_extras {
    ($type_name: ident $(, $l: lifetime )?) => {
        impl<$($l,)?T> std::ops::Index<usize> for $type_name<$($l,)?T> {
            type Output = T;
            #[inline]
            fn index(&self, index: usize) -> &Self::Output {
                self.as_ref().index(index)
            }
        }

        impl<$($l,)?T> PartialEq<Tensor<T>> for $type_name<$($l,)?T> where T: PartialEq {
            fn eq(&self, other: &Tensor<T>) -> bool {
                self.dims() == other.dims() && self.as_ref() == other.as_ref()
            }
        }

        impl<$($l,)?'b, T> PartialEq<TensorView<'b, T>> for $type_name<$($l,)?T> where T: PartialEq {
            fn eq(&self, other: &TensorView<'b, T>) -> bool {
                self.dims() == other.dims() && self.as_ref() == other.as_ref()
            }
        }

        impl<$($l,)?'b, T> PartialEq<TensorViewMut<'b, T>> for $type_name<$($l,)?T> where T: PartialEq {
            fn eq(&self, other: &TensorViewMut<'b, T>) -> bool {
                self.dims() == other.dims() && self.as_ref() == other.as_ref()
            }
        }

        impl<$($l,)?'b, T> PartialEq<TensorCow<'b, T>> for $type_name<$($l,)?T> where T: PartialEq {
            fn eq(&self, other: &TensorCow<'b, T>) -> bool {
                self.dims() == other.dims() && self.as_ref() == other.as_ref()
            }
        }

    };
}

macro_rules! impl_tensor_extras_mut {
    ($type_name: ident $(, $l: lifetime )?) => {
        impl<$($l,)?T> std::ops::IndexMut<usize> for $type_name<$($l,)?T> {
            #[inline]
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                self.as_mut().index_mut(index)
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

