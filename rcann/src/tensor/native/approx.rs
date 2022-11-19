use std::iter::zip;
use approx::{AbsDiffEq};
use crate::tensor::{Dims, ITensor, Tensor, TensorView, TensorViewMut, TensorCow};

macro_rules! impl_tensor_approx {
    ($type_name: ident $(, $l: lifetime )?) => {
        impl<$($l,)?T: AbsDiffEq, D: Dims> AbsDiffEq<Tensor<T, D>> for $type_name<$($l,)?T, D> where T::Epsilon : Copy {
            type Epsilon = T::Epsilon;
            fn default_epsilon() -> Self::Epsilon {
                T::default_epsilon()
            }
            fn abs_diff_eq(&self, other: &Tensor<T, D>, epsilon: Self::Epsilon) -> bool {
                self.dims() == other.dims() && zip(self.as_ref(), other.as_ref())
                    .all(|(a, b)| T::abs_diff_eq(a, b, epsilon))
            }
        }
        impl<$($l,)?'b, T: AbsDiffEq, D: Dims> AbsDiffEq<TensorView<'b, T, D>> for $type_name<$($l,)?T, D> where T::Epsilon : Copy {
            type Epsilon = T::Epsilon;
            fn default_epsilon() -> Self::Epsilon {
                T::default_epsilon()
            }
            fn abs_diff_eq(&self, other: &TensorView<'b, T, D>, epsilon: Self::Epsilon) -> bool {
                self.dims() == other.dims() && zip(self.as_ref(), other.as_ref())
                    .all(|(a, b)| T::abs_diff_eq(a, b, epsilon))
            }
        }
        impl<$($l,)?'b, T: AbsDiffEq, D: Dims> AbsDiffEq<TensorViewMut<'b, T, D>> for $type_name<$($l,)?T, D> where T::Epsilon : Copy {
            type Epsilon = T::Epsilon;
            fn default_epsilon() -> Self::Epsilon {
                T::default_epsilon()
            }
            fn abs_diff_eq(&self, other: &TensorViewMut<'b, T, D>, epsilon: Self::Epsilon) -> bool {
                self.dims() == other.dims() && zip(self.as_ref(), other.as_ref())
                    .all(|(a, b)| T::abs_diff_eq(a, b, epsilon))
            }
        }
        impl<$($l,)?'b, T: AbsDiffEq, D: Dims> AbsDiffEq<TensorCow<'b, T, D>> for $type_name<$($l,)?T, D> where T::Epsilon : Copy {
            type Epsilon = T::Epsilon;
            fn default_epsilon() -> Self::Epsilon {
                T::default_epsilon()
            }
            fn abs_diff_eq(&self, other: &TensorCow<'b, T, D>, epsilon: Self::Epsilon) -> bool {
                self.dims() == other.dims() && zip(self.as_ref(), other.as_ref())
                    .all(|(a, b)| T::abs_diff_eq(a, b, epsilon))
            }
        }
    };
}

impl_tensor_approx!(Tensor);
impl_tensor_approx!(TensorView, 'a);
impl_tensor_approx!(TensorViewMut, 'a);
impl_tensor_approx!(TensorCow, 'a);


