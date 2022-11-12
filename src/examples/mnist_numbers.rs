use std::iter::zip;
use std::time::Duration;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use crate::activation::ActivationFn;
use crate::backend::CpuBackend;
use crate::loss::LossFn;
use crate::net::initializer::RandomNetInitializer;
use crate::net::layer::FullyConnectedLayerParams;
use crate::net::{NetBuilder, TrainBatchResult};
use crate::tensor::{ITensorBase, Tensor, TensorBase, TensorView};
use crate::util::max_index;
use super::util::{MnistData, load_mnist_data};

pub fn train_minst() {
    
    let MnistData { mut train, mut test } = load_mnist_data::<f32>(60_000, 10_000, 64);

    let mut shuffle_rng = StdRng::seed_from_u64(0xf666);

    let mut net = NetBuilder::new(CpuBackend::<f32>::new(), 784)
        .with_initializer(RandomNetInitializer::seed_from_u64(0xf1234567))
        .with_layer(FullyConnectedLayerParams { size: 128, activation_fn: ActivationFn::Sigmoid })
        .with_layer(FullyConnectedLayerParams { size: 32, activation_fn: ActivationFn::Sigmoid })
        .with_layer(FullyConnectedLayerParams { size: 10, activation_fn: ActivationFn::Softmax })
        .build()
        .unwrap();

    //println!("{:#?}", net);

    let max_epochs = 100;

    for epoch in 0..max_epochs {

        train.shuffle(&mut shuffle_rng);

        let mut rmse: f32 = 0.0;
        let mut mse: f32 = 0.0;
        let mut wrong: usize = 0;
        let mut total: usize = 0;

        for (image_data, labels) in train.iter() {
            let TrainBatchResult { error, output } = net.train_batch(image_data, labels, &LossFn::MSE, 0.1, 0.1);
            total += error.len();
            mse += error.iter().sum::<f32>();
            rmse += error.iter().map(|&x| x.sqrt()).sum::<f32>();
            wrong += count_batch_errors(labels, output);
        }

        mse /= total as f32;
        rmse /= total as f32;

        println!("epoch: {epoch}, avg RMSE: {rmse}, avg MSE: {mse}, error rate: {}%", (wrong as f32 / total as f32) * 100.0);

    }

    //println!("{:#?}", net);

    let mut total = 0;
    let mut errors = 0;
    for (image_data, labels) in test.iter() {
        let output = net.predict(image_data);
        errors += count_batch_errors(labels, output);
        total += output.dims().first();
    }

    println!("test error rate: {}", (errors as f32 / total as f32) * 100.0);

}

fn count_batch_errors(labels: &Tensor<f32>, output: &Tensor<f32>) -> usize {
    zip(labels.iter_first_axis(), output.iter_first_axis())
        .fold(0, |sum, (l, o)| {
            if max_index(&l) == max_index(&o) {
                sum
            } else {
                sum + 1
            }
        })
}