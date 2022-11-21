use rand::prelude::Distribution;
use rand::{SeedableRng};
use rand::rngs::StdRng;
use rand_distr::StandardNormal;
use crate::tensor::{Dim2, Tensor2};

pub const MATRIX_SIZE_LG: usize = 2048;
pub const MATRIX_SIZE_MD: usize = 512;
pub const MATRIX_SIZE_SM: usize = 128;
pub const SEED: u64 = 0x8371943;

pub fn get_square_matmul_tensors<T>(size: usize) -> [Tensor2<T>; 3] where StandardNormal: Distribution<T> {
    let mut rng = StdRng::seed_from_u64(SEED);
    [
        Tensor2::from_distribution(&mut rng, StandardNormal, Dim2(size, size)),
        Tensor2::from_distribution(&mut rng, StandardNormal, Dim2(size, size)),
        Tensor2::from_distribution(&mut rng, StandardNormal, Dim2(size, size)),
    ]
}
