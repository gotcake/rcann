mod dims;
mod native;

use std::fmt::{Debug, Formatter};
pub use dims::*;
pub use native::base::*;
pub use native::iter::*;
pub use native::owned::*;
pub use native::view::*;
pub use native::cow::*;

/// Generic Tensor type which is specialized by different backends
pub trait ITensor<T> {
    fn len(&self) -> usize;
    fn dims(&self) -> &Dims;
}
