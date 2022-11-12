use rand::{Rng, SeedableRng};
use rand::distributions::Distribution;
use rand::rngs::StdRng;
use rand_distr::Normal;
use crate::dtype::DType;
use crate::net::layer::LayerType;
use crate::tensor::{Dims, Tensor};

pub trait NetInitializer<T: DType> {
    fn get_weights(
        &mut self,
        layer_type: LayerType,
        dim: Dims,
        layer_idx: usize,
        input_size: usize,
        output_size: usize,
    ) -> Tensor<T>;
    fn get_biases(
        &mut self,
        layer_type: LayerType,
        dims: Dims,
        layer_idx: usize,
    ) -> Tensor<T>;
}

pub struct RandomNetInitializer {
    rng: StdRng,
}

impl RandomNetInitializer {
    pub fn seed_from_u64(seed: u64) -> Self {
        RandomNetInitializer {
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl Default for RandomNetInitializer {
    fn default() -> Self {
        RandomNetInitializer {
            rng: StdRng::from_entropy()
        }
    }
}

impl<T: DType> NetInitializer<T> for RandomNetInitializer {

    fn get_weights(&mut self, layer_type: LayerType, dim: Dims, layer_idx: usize, input_size: usize, output_size: usize) -> Tensor<T> {
        let std = (2.0 / (input_size + output_size) as f64).sqrt();
        let dist = Normal::new(0.0, std).unwrap();
        let vec: Vec<T> = dist.sample_iter(&mut self.rng)
            .take(dim.tensor_len())
            .map(T::from_f64)
            .collect();
        Tensor::from_vec(vec, dim)
    }

    fn get_biases(&mut self, layer_type: LayerType, dims: Dims, layer_idx: usize) -> Tensor<T> {
        Tensor::zero(dims)
    }

}