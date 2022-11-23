use std::ops::Neg;
use num_traits::{Float, FloatConst, Num, NumAssignOps, NumCast, NumOps, One, PrimInt, Zero};


pub unsafe trait DType: 'static + Sized + Copy + Num + NumAssignOps {
    const ZERO: Self;
    const ONE: Self;
    fn from_f64(val: f64) -> Self;
    fn from_usize(val: usize) -> Self;
}

pub unsafe trait DTypeUInt: DType {}
pub unsafe trait DTypeSInt: DType + Neg<Output=Self> {}
pub unsafe trait DTypeFloat: DType + Float {}

macro_rules! impl_dtype {
    ($ty:ty, $one:expr, $zero:expr $(,$other_trait:ty)*) => {
        unsafe impl DType for $ty {
            const ZERO: Self = $zero;
            const ONE: Self = $one;
            #[inline]
            fn from_f64(val: f64) -> Self {
                val as $ty
            }
            #[inline]
            fn from_usize(val: usize) -> Self {
                val as $ty
            }
        }
        $(
        unsafe impl $other_trait for $ty {}
        )*
    };
}

impl_dtype!(f32, 1.0, 0.0, DTypeFloat);
impl_dtype!(f64, 1.0, 0.0, DTypeFloat);
impl_dtype!(i8, 1, 0, DTypeSInt);
impl_dtype!(i32, 1, 0, DTypeSInt);
impl_dtype!(i64, 1, 0, DTypeSInt);
impl_dtype!(u8, 1, 0, DTypeUInt);
impl_dtype!(u16, 1, 0, DTypeUInt);
impl_dtype!(u32, 1, 0, DTypeUInt);
impl_dtype!(u64, 1, 0, DTypeUInt);
impl_dtype!(usize, 1, 0, DTypeUInt);




