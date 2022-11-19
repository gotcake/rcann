use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rcann::activation::ActivationFn;
use rcann::backend::CpuBackend;
use rcann::dtype::DType;
use rcann::loss::LossFn;
use rcann::net::initializer::RandomNetInitializer;
use rcann::net::layer::FullyConnectedLayerParams;
use rcann::net::{NetBuilder, TrainBatchResult};
use rcann::tensor::{ITensorBase, Tensor, TensorBase};
use std::iter::zip;

use rcann_examples::util::{load_mnist_data, max_index, MnistData};

pub fn main() {
    let MnistData { mut train, test } = load_mnist_data::<f64>(60_000, 10_000, 64);

    let mut shuffle_rng = StdRng::seed_from_u64(0xf666);

    let mut net = NetBuilder::new(CpuBackend::<f64>::new(), 784)
        .with_initializer(RandomNetInitializer::seed_from_u64(0xf1234567))
        .with_layer(FullyConnectedLayerParams {
            size: 128,
            activation_fn: ActivationFn::Sigmoid,
        })
        .with_layer(FullyConnectedLayerParams {
            size: 32,
            activation_fn: ActivationFn::Sigmoid,
        })
        .with_layer(FullyConnectedLayerParams {
            size: 10,
            activation_fn: ActivationFn::Softmax,
        })
        .build()
        .unwrap();

    //println!("{:#?}", net);

    let max_epochs = 100;

    for epoch in 0..max_epochs {
        train.shuffle(&mut shuffle_rng);

        let mut rmse: f64 = 0.0;
        let mut mse: f64 = 0.0;
        let mut wrong: usize = 0;
        let mut total: usize = 0;

        for (image_data, labels) in train.iter() {
            let TrainBatchResult { error, output } =
                net.train_batch(image_data, labels, &LossFn::MSE, 0.1, 0.1);
            total += error.len();
            mse += error.iter().sum::<f64>();
            rmse += error.iter().map(|&x| x.sqrt()).sum::<f64>();
            wrong += count_batch_errors(labels, output);
        }

        mse /= total as f64;
        rmse /= total as f64;

        println!(
            "epoch: {epoch}, avg RMSE: {rmse}, avg MSE: {mse}, error rate: {}%",
            (wrong as f32 / total as f32) * 100.0
        );
    }

    //println!("{:#?}", net);

    let mut total = 0;
    let mut errors = 0;
    for (image_data, labels) in test.iter() {
        let output = net.predict(image_data);
        errors += count_batch_errors(labels, output);
        total += output.dims().first();
    }

    println!(
        "test error rate: {}",
        (errors as f32 / total as f32) * 100.0
    );
}

fn count_batch_errors<T: DType>(labels: &Tensor<T>, output: &Tensor<T>) -> usize {
    zip(labels.iter_first_axis(), output.iter_first_axis()).fold(0, |sum, (l, o)| {
        if max_index(&l) == max_index(&o) {
            sum
        } else {
            sum + 1
        }
    })
}
