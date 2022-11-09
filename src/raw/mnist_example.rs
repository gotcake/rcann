use std::cmp::Ordering;
use std::fs::File;
use std::io::BufWriter;
use std::iter::zip;
use std::time::SystemTime;
use mnist::{Mnist, MnistBuilder};
use rand::rngs::StdRng;
use rand::SeedableRng;
use crate::activation::ActivationFn;
use crate::loss::LossFn;
use crate::raw::net::{Layer, Net};

pub fn train_mnist() {

    let n_train_samples: usize = 60_000;
    let n_tst_samples: usize = 10000;
    let n_pixels: usize = 28 * 28;
    let n_classes: usize = 10;

    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_one_hot()
        .training_set_length(n_train_samples as u32)
        .test_set_length(n_tst_samples as u32)
        .finalize();

    assert_eq!(trn_img.len(), n_train_samples * n_pixels);
    assert_eq!(trn_lbl.len(), n_train_samples * n_classes);
    assert_eq!(tst_img.len(), n_tst_samples * n_pixels);
    assert_eq!(tst_lbl.len(), n_tst_samples * n_classes);

    let inputs: Vec<f32> = trn_img.into_iter()
        .map(|p| p as f32 / 256.0)
        .collect();

    let expected: Vec<f32> = trn_lbl.into_iter()
        .map(|x| x as f32)
        .collect();

    let mut rng = StdRng::seed_from_u64(0xf1234567);

    let mut net = Net::new(
        Layer::new(n_pixels as usize, 512, ActivationFn::Sigmoid, &mut rng),
        vec![
            Layer::new(512, 256, ActivationFn::Sigmoid, &mut rng),
            Layer::new(256, 128, ActivationFn::Sigmoid, &mut rng),
            Layer::new(128, 64, ActivationFn::Sigmoid, &mut rng),
        ],
        Layer::new(64, 10, ActivationFn::Softmax, &mut rng),
    );

    println!("{:?}", net);

    let start = SystemTime::now();

    net.train(
        &inputs,
        &expected,
        &LossFn::MSE,
        64,
        0.1,
        0.1,
        100,
        &mut rng,
    );

    let elapsed = start.elapsed().unwrap();

    println!("Elapsed time: {} seconds", elapsed.as_secs());
    println!("{:?}", net);

    let test_inputs: Vec<f32> = tst_img.into_iter()
        .map(|p| p as f32 / 256.0)
        .collect();

    let mut total: usize = 0;
    let mut errors: usize = 0;
    for (t_input, t_lbl) in zip(test_inputs.chunks_exact(net.input_len()), tst_lbl.chunks_exact(net.output_len())) {
        let output = net.forward(1, t_input);
        let estimate = argmax(output);
        let actual = argmax(t_lbl);

        if total % 100 == 0 {
            println!("actual: {}, estimate: {}, output: {:?}", actual, estimate, output);
        }
        total += 1;
        if estimate != actual {
            errors += 1;
        }
    }

    println!("Test errors: {}/{} ({:.3}%)", errors, total, (errors as f32 / total as f32) * 100.);

    let f = File::create("data/mnist_net.json").unwrap();
    let writer = BufWriter::new(f);
    serde_json::to_writer_pretty(writer, &net).unwrap();

}

fn argmax<T: PartialOrd>(a: &[T]) -> usize {
    a.iter()
        .enumerate()
        .max_by(|&i, &j| if i.1 < j.1 { Ordering::Less } else { Ordering::Greater })
        .map(|i| i.0)
        .unwrap()
}