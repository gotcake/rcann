use crate::dtype::DType;
use crate::net::layer::LayerType;
use rand::distributions::Distribution;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::Normal;

pub trait NetInitializer<T: DType> {
    fn get_weights(
        &mut self,
        layer_type: LayerType,
        count: usize,
        layer_idx: usize,
        input_size: usize,
        output_size: usize,
    ) -> Vec<T>;
    fn get_biases(&mut self, layer_type: LayerType, count: usize, layer_idx: usize) -> Vec<T>;
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
            rng: StdRng::from_entropy(),
        }
    }
}

impl<T: DType> NetInitializer<T> for RandomNetInitializer {
    fn get_weights(
        &mut self,
        _layer_type: LayerType,
        count: usize,
        _layer_idx: usize,
        input_size: usize,
        output_size: usize,
    ) -> Vec<T> {
        let std = (2.0 / (input_size + output_size) as f64).sqrt();
        let dist = Normal::new(0.0, std).unwrap();
        dist.sample_iter(&mut self.rng)
            .take(count)
            .map(T::from_f64)
            .collect()
    }

    fn get_biases(&mut self, _layer_type: LayerType, count: usize, _layer_idx: usize) -> Vec<T> {
        vec![T::ZERO; count]
    }
}
