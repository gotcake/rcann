mod activation;
mod loss;
mod data;
mod raw;
mod hpo;
mod tensor;
mod dtype;
mod backend;
mod cpu;
pub mod net;

extern crate num_traits;
extern crate matrixmultiply;
extern crate rand;
extern crate rand_distr;
extern crate serde;
extern crate serde_json;
extern crate base64;

fn main() {
    crate::raw::mnist_example::train_mnist();
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


