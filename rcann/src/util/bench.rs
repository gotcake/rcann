use crate::tensor::{Dim2, Tensor2};
use rand::prelude::Distribution;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::StandardNormal;

pub const SIZE_LG: usize = 2048;
pub const SIZE_MD: usize = 512;
pub const SIZE_SM: usize = 128;
const SEED: u64 = 0x8371943;

pub fn get_square_matrices<T>(size: usize) -> [Tensor2<T>; 3]
where
    StandardNormal: Distribution<T>,
{
    let mut rng = StdRng::seed_from_u64(SEED);
    [
        Tensor2::from_distribution(&mut rng, StandardNormal, Dim2(size, size)),
        Tensor2::from_distribution(&mut rng, StandardNormal, Dim2(size, size)),
        Tensor2::from_distribution(&mut rng, StandardNormal, Dim2(size, size)),
    ]
}
