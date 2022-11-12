#![feature(slice_split_at_unchecked)]

pub mod activation;
pub mod backend;
pub mod dtype;
pub mod loss;
pub mod net;
pub mod tensor;

extern crate base64;
extern crate matrixmultiply;
extern crate num_traits;
extern crate rand;
extern crate rand_distr;
extern crate serde;
extern crate serde_json;
