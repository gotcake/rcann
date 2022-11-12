use std::fmt::{LowerExp};
use std::ops::Neg;
use num_traits::{Float, NumAssignOps, NumOps, One, Zero, FloatConst, NumCast};

pub trait DType: 'static + Float + FloatConst + NumOps + NumAssignOps + One + Zero + Neg + NumCast + LowerExp {
    const ZERO: Self;
    const ONE: Self;
    fn from_f64(val: f64) -> Self;
    fn from_usize(val: usize) -> Self;
}
impl DType for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    #[inline]
    fn from_f64(val: f64) -> Self {
        val as f32
    }
    #[inline]
    fn from_usize(val: usize) -> Self {
        val as f32
    }
}
impl DType for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    #[inline]
    fn from_f64(val: f64) -> Self {
        val
    }
    #[inline]
    fn from_usize(val: usize) -> Self {
        val as f64
    }
}