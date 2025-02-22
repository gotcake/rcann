use num_traits::{Float, Num, NumAssignOps};
use std::fmt::{Debug, Display};
use std::ops::Neg;

pub unsafe trait DType: 'static + Sized + Copy + Num + NumAssignOps + Display + Debug {
    const ZERO: Self;
    const ONE: Self;
    const BITS: u8;
    fn from_f64(val: f64) -> Self;
    fn from_usize(val: usize) -> Self;
    fn to_usize(&self) -> usize;
    fn to_f64(&self) -> f64;
}

pub unsafe trait DTypeUInt: DType {}
pub unsafe trait DTypeSInt: DType + Neg<Output = Self> {}
pub unsafe trait DTypeFloat: DType + Float {}

macro_rules! impl_dtype {
    ($ty:ty, $one:expr, $zero:expr $(,$other_trait:ty)*) => {
        unsafe impl DType for $ty {
            const ZERO: Self = $zero;
            const ONE: Self = $one;
            const BITS: u8 = (std::mem::size_of::<$ty>() * 8) as u8;
            #[inline]
            fn from_f64(val: f64) -> Self {
                val as $ty
            }
            #[inline]
            fn from_usize(val: usize) -> Self {
                val as $ty
            }
            #[inline]
            fn to_f64(&self) -> f64 {
                *self as f64
            }
            #[inline]
            fn to_usize(&self) -> usize {
                *self as usize
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

#[cfg(feature = "half")]
unsafe impl DType for half::f16 {
    const ZERO: Self = half::f16::ZERO;
    const ONE: Self = half::f16::ONE;
    const BITS: u8 = 16;

    #[inline]
    fn from_f64(val: f64) -> Self {
        half::f16::from_f64(val)
    }

    #[inline]
    fn from_usize(val: usize) -> Self {
        half::f16::from_f64(val as f64)
    }

    #[inline]
    fn to_usize(&self) -> usize {
        half::f16::to_f64(*self) as usize
    }

    #[inline]
    fn to_f64(&self) -> f64 {
        half::f16::to_f64(*self)
    }
}

#[cfg(feature = "half")]
unsafe impl DTypeFloat for half::f16 {}