use mnist::{Mnist, MnistBuilder};
use rcann::dtype::DTypeFloat;
use rcann::tensor::{Dim2, Tensor2};

const IMAGE_PIXELS: usize = 28 * 28;
const NUM_CLASSES: usize = 10;

pub struct MnistData<D: DTypeFloat> {
    pub train_images: Tensor2<D>,
    pub train_labels: Tensor2<D>,
    pub test_images: Tensor2<D>,
    pub test_labels: Tensor2<D>,
}

pub fn load_mnist_data<D: DTypeFloat>(train_samples: usize, test_samples: usize) -> MnistData<D> {
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
        train_images: Tensor2::from_vec(
            trn_img.into_iter().map(|p| D::from_f64(p as f64 / 256.0)).collect(),
            Dim2(train_samples, IMAGE_PIXELS),
        ),
        train_labels: Tensor2::from_vec(
            trn_lbl.into_iter().map(|l| D::from_usize(l as usize)).collect(),
            Dim2(train_samples, NUM_CLASSES),
        ),
        test_images: Tensor2::from_vec(
            tst_img.into_iter().map(|p| D::from_f64(p as f64 / 256.0)).collect(),
            Dim2(test_samples, IMAGE_PIXELS),
        ),
        test_labels: Tensor2::from_vec(
            tst_lbl.into_iter().map(|l| D::from_usize(l as usize)).collect(),
            Dim2(test_samples, NUM_CLASSES),
        ),
    }
}
