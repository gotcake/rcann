use std::ops::Neg;
use num_traits::{Float, NumAssignOps, NumOps, One, Zero, FloatConst};

pub trait DType: Float + FloatConst + NumOps + NumAssignOps + One + Zero + Neg {
    fn from_f64(val: f64) -> Self;
}
impl DType for f32 {
    #[inline]
    fn from_f64(val: f64) -> Self {
        val as f32
    }
}
impl DType for f64 {
    #[inline]
    fn from_f64(val: f64) -> Self {
        val
    }
}