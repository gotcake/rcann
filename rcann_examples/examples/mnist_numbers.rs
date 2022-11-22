use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rcann::activation::ActivationFn;
use rcann::backend::{BackendOther, CpuBackend, TensorOps};
use rcann::dtype::DType;
use rcann::loss::LossFn;
use rcann::net::initializer::RandomNetInitializer;
use rcann::net::layer::FullyConnectedLayerParams;
use rcann::net::{NetBuilder, TrainBatchResult};
use rcann::tensor::{Dims, ITensor, Tensor2, TensorBase};
use std::iter::zip;
use std::time::Instant;

use rcann_examples::util::{load_mnist_data, max_index, MnistData};
use rcann_opencl::backend::OpenCLBackend;
use rcann_opencl::tensor::{OclTensor, OclTensor2};

const MAX_BATCH_SIZE: usize = 64;

pub fn main() {
    let MnistData { mut train, test } = load_mnist_data::<f32>(60_000, 10_000, MAX_BATCH_SIZE);

    let mut shuffle_rng = StdRng::seed_from_u64(0xf666);

    let backend = OpenCLBackend::from_default_device(MAX_BATCH_SIZE).unwrap();
    //let backend = CpuBackend::<f32>::new(MAX_BATCH_SIZE);

    let mut net = NetBuilder::new(backend, 784)
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
            activation_fn: ActivationFn::Sigmoid,
        })
        .build()
        .unwrap();

    /*let mut train: Vec<_> = train.into_iter().map(|(image_data, labels)| {
        (net.backend().new_tensor_from_native(image_data.view()), net.backend().new_tensor_from_native(labels.view()), image_data, labels)
    }).collect();*/

    println!("{:#?}", net);

    let max_epochs = 100;
    let start = Instant::now();

    for epoch in 0..max_epochs {
        train.shuffle(&mut shuffle_rng);

        let mut rmse: f32 = 0.0;
        let mut mse: f32 = 0.0;
        let mut wrong: usize = 0;
        let mut total: usize = 0;

        for (image_data_native, labels_native) in train.iter() {
            let TrainBatchResult { output, error } = net.train_batch(image_data_native, labels_native, &LossFn::MSE, 0.1, 0.1);
            total += error.len();
            mse += error.iter().sum::<f32>();
            rmse += error.iter().map(|&x| x.sqrt()).sum::<f32>();
            wrong += count_batch_errors(labels_native, output);
        }

        mse /= total as f32;
        rmse /= total as f32;


        println!(
            "epoch: {epoch}, avg RMSE: {rmse}, avg MSE: {mse}, error rate: {}%",
            (wrong as f32 / total as f32) * 100.0
        );
    }

    let elapsed = start.elapsed();
    println!(
        "Training time for {max_epochs} epochs and batch size {MAX_BATCH_SIZE}: {} sec",
        elapsed.as_secs_f32()
    );

    let mut total = 0;
    let mut errors = 0;
    for (image_data, labels) in test.iter() {
        let output = net.predict(image_data);
        errors += count_batch_errors(labels, output);
        total += output.dims().major();
    }

    println!("test error rate: {}", (errors as f32 / total as f32) * 100.0);
}

fn count_batch_errors<T: DType>(labels: &Tensor2<T>, output: &Tensor2<T>) -> usize {
    zip(labels.iter_major_axis(), output.iter_major_axis()).fold(0, |sum, (l, o)| {
        if max_index(l.as_ref()) == max_index(o.as_ref()) {
            sum
        } else {
            sum + 1
        }
    })
}
