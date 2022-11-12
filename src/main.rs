#![feature(slice_split_at_unchecked)]

mod activation;
mod loss;
mod data;
mod raw;
mod hpo;
mod dtype;
mod backend;
pub mod net;
mod util;
mod tensor;
mod examples;

extern crate num_traits;
extern crate matrixmultiply;
extern crate rand;
extern crate rand_distr;
extern crate serde;
extern crate serde_json;
extern crate base64;

fn main() {
    crate::examples::mnist_numbers::train_minst();
}
/*
fn train_xor() {

    let inputs = vec![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let expected = vec![[0.0], [1.0], [1.0], [0.0]];

    let mut net = Net::new(
        Layer::new(2, 3, ActivationFn::Sigmoid),
        vec![
            Layer::new(3, 3, ActivationFn::Sigmoid),
        ],
        Layer::new(3, 1, ActivationFn::Sigmoid),
    );

    println!("{:?}", net);

    net.train(
        &inputs,
        &expected,
        &LossFn::MSE,
        4,
        0.1,
        10000,
    );

    println!("{:?}", net);

}
*/


