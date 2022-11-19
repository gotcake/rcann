mod dims;
mod native;

pub use dims::*;
pub use native::base::*;
pub use native::cow::*;
pub use native::iter::*;
pub use native::owned::*;
pub use native::view::*;

/// Generic Tensor type which is specialized by different backends
pub trait ITensor<T, D: Dims> {
    fn len(&self) -> usize;
    fn dims(&self) -> &D;
}
