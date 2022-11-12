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
use crate::tensor::{Tensor, TensorBase, TensorView};
use super::util::{MnistData, load_mnist_data};

pub fn train_minst() {
    
    let MnistData { mut train, mut test } = load_mnist_data::<f32>(60_000, 10_000, 64);

    let mut shuffle_rng = StdRng::seed_from_u64(0xf666);

    let mut net = NetBuilder::new(CpuBackend::<f32>::new(), 784)
        .with_initializer(RandomNetInitializer::seed_from_u64(0xf1234567))
        .with_layer(FullyConnectedLayerParams { size: 128, activation_fn: ActivationFn::Sigmoid })
        .with_layer(FullyConnectedLayerParams { size: 32, activation_fn: ActivationFn::Sigmoid })
        .with_layer(FullyConnectedLayerParams { size: 10, activation_fn: ActivationFn::Sigmoid })
        .build()
        .unwrap();

    println!("{:#?}", net);

    let max_epochs = 10;

    for epoch in 0..max_epochs {

        train.shuffle(&mut shuffle_rng);

        let mut rmse: f32 = 0.0;
        let mut mse: f32 = 0.0;
        let mut correct: usize = 0;
        let mut total: usize = 0;

        for (image_data, labels) in train.iter() {
            let TrainBatchResult { error, output } = net.train_batch(image_data, labels, &LossFn::MSE, 0.1, 0.1);
            total += error.len();
            let b_mse = error.iter().sum::<f32>();
            let b_rmse = error.iter().map(|&x| x.sqrt()).sum::<f32>();
            mse += b_mse;
            rmse += b_rmse;
            println!("avg RMSE: {}, avg MSE: {}, output: {output:?}", b_rmse / error.len() as f32, b_mse / error.len() as f32);
            println!("{:#?}", net);
            std::thread::sleep(Duration::from_millis(1000));
            //break;
        }

        mse /= total as f32;
        rmse /= total as f32;

        println!("epoch: {epoch}, avg RMSE: {rmse}, avg MSE: {mse}");
        break;

    }

    println!("{:#?}", net);

    
}