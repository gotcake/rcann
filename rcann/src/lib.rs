#![feature(slice_split_at_unchecked)]

pub mod activation;
pub mod loss;
mod data;
mod raw;
mod hpo;
pub mod dtype;
pub mod backend;
pub mod net;
mod util;
pub mod tensor;

extern crate num_traits;
extern crate matrixmultiply;
extern crate rand;
extern crate rand_distr;
extern crate serde;
extern crate serde_json;
extern crate base64;


