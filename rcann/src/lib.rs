//#![feature(slice_split_at_unchecked)]

pub mod activation;
pub mod backend;
pub mod dtype;
pub mod loss;
pub mod net;
pub mod tensor;
mod util;

extern crate matrixmultiply;
extern crate num_traits;
extern crate rand;
extern crate rand_distr;
#[cfg(feature = "serde")]
extern crate serde;
#[cfg(feature = "serde")]
extern crate serde_json;
#[cfg(feature = "approx")]
extern crate approx;
