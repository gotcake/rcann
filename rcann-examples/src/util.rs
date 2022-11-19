use mnist::{Mnist, MnistBuilder};
use rcann::dtype::DType;
use rcann::tensor::{Dim2, Tensor2};
use std::cmp::Ordering;
use std::iter::zip;

const IMAGE_PIXELS: usize = 28 * 28;
const NUM_CLASSES: usize = 10;

pub struct MnistData<D: DType> {
    pub train: Vec<(Tensor2<D>, Tensor2<D>)>,
    pub test: Vec<(Tensor2<D>, Tensor2<D>)>,
}

pub fn load_mnist_data<D: DType>(
    train_samples: usize,
    test_samples: usize,
    batch_size: usize,
) -> MnistData<D> {
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .base_path("rcann-examples/data")
        .label_format_one_hot()
        .training_set_length(train_samples as u32)
        .test_set_length(test_samples as u32)
        .finalize();

    assert_eq!(trn_img.len(), train_samples * IMAGE_PIXELS);
    assert_eq!(trn_lbl.len(), train_samples * NUM_CLASSES);
    assert_eq!(tst_img.len(), test_samples * IMAGE_PIXELS);
    assert_eq!(tst_lbl.len(), test_samples * NUM_CLASSES);

    MnistData {
        train: zip(
            get_batches(trn_img, IMAGE_PIXELS, batch_size, |p| {
                D::from_f64(p as f64 / 256.0)
            }),
            get_batches(trn_lbl, NUM_CLASSES, batch_size, |l| {
                D::from_usize(l as usize)
            }),
        )
        .collect(),
        test: zip(
            get_batches(tst_img, IMAGE_PIXELS, batch_size, |p| {
                D::from_f64(p as f64 / 256.0)
            }),
            get_batches(tst_lbl, NUM_CLASSES, batch_size, |l| {
                D::from_usize(l as usize)
            }),
        )
        .collect(),
    }
}

fn get_batches<I: Copy, O, F>(
    raw: Vec<I>,
    sample_size: usize,
    batch_size: usize,
    f: F,
) -> Vec<Tensor2<O>>
where
    F: Fn(I) -> O,
{
    assert_eq!(raw.len() % sample_size, 0);
    raw.chunks(sample_size * batch_size)
        .map(|chunk| {
            let num_samples = chunk.len() / sample_size;
            let converted: Vec<O> = chunk.iter().map(|x| f(*x)).collect();
            Tensor2::from_vec(converted, Dim2(num_samples, sample_size))
        })
        .collect()
}

pub fn max_index<T: Copy + PartialOrd>(a: &[T]) -> usize {
    a.iter()
        .enumerate()
        .max_by(|&(_, &a), &(_, &b)| {
            if a < b {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        })
        .expect("expected at least one element")
        .0
}
