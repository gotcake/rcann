extern crate mnist;
extern crate rand;

mod mnist_numbers;
mod util;

fn main() {
    mnist_numbers::train_minst();
}
