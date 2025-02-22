use rand::rngs::StdRng;
use rand::SeedableRng;
use rcann::activation::ActivationFn;
use rcann::backend::CpuBackend;
use rcann::net::initializer::RandomNetInitializer;
use rcann::net::layer::DenseLayerParams;
use rcann::net::NetBuilder;
use rcann::scoring::MulticlassScorer;
use rcann::tensor::TensorBase;
use rcann_examples::util::{load_mnist_data, MnistData};
use rcann_opencl::backend::OpenCLBackend;
use std::time::Instant;

const MAX_BATCH_SIZE: usize = 64;

pub fn main() {
    let MnistData {
        train_images,
        train_labels,
        test_images,
        test_labels,
    } = load_mnist_data::<f32>(60_000, 10_000);

    let mut shuffle_rng = StdRng::seed_from_u64(0xf666);

    let backend = OpenCLBackend::from_default_device(MAX_BATCH_SIZE).unwrap();
    //let backend = CpuBackend::<f32>::new(MAX_BATCH_SIZE);

    let mut net = NetBuilder::new(backend, 784)
        .with_initializer(RandomNetInitializer::seed_from_u64(0xf1234567))
        .with_layer(DenseLayerParams {
            size: 128,
            activation_fn: ActivationFn::Sigmoid,
        })
        .with_layer(DenseLayerParams {
            size: 32,
            activation_fn: ActivationFn::Sigmoid,
        })
        .with_layer(DenseLayerParams {
            size: 10,
            activation_fn: ActivationFn::Sigmoid,
        })
        .build()
        .unwrap();

    //println!("{:#?}", net);

    let max_epochs = 100;
    let start = Instant::now();

    net.train(&mut shuffle_rng, train_images.view(), train_labels.view(), max_epochs);

    let elapsed = start.elapsed();
    println!(
        "Training time for {max_epochs} epochs and batch size {MAX_BATCH_SIZE}: {} sec",
        elapsed.as_secs_f32()
    );

    let mut scorer = MulticlassScorer::for_net(&net);
    net.evaluate(test_images.view(), test_labels.view(), &mut scorer);

    scorer.print_report(net.backend());
}
