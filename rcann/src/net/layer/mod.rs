mod concrete;
mod fully_connected;

use crate::backend::Backend;
use crate::net::initializer::NetInitializer;
use std::fmt::Debug;

use crate::tensor::Dim2;
pub use concrete::{ConcreteLayer, ConcreteLayerParams};
pub use fully_connected::{DenseLayer, DenseLayerParams};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum LayerType {
    FullyConnected,
}

pub trait LayerParams<B: Backend>: Clone + Debug {
    type Layer: Layer<B>;

    fn create_layer(
        &self,
        backend: &B,
        layer_idx: usize,
        input_size: usize,
        initializer: &mut dyn NetInitializer<B::Float>,
    ) -> Self::Layer;
}

pub trait Layer<B: Backend>: Debug {
    fn forward(&mut self, backend: &B, input: B::TensorRef<'_, Dim2>, output: &mut B::Tensor<Dim2>);

    fn backprop(
        &mut self,
        backend: &B,
        input: B::TensorRef<'_, Dim2>,
        output: &B::Tensor<Dim2>,
        input_error: Option<&mut B::Tensor<Dim2>>,
        output_error: &B::Tensor<Dim2>,
        learn_rate: B::Float,
        momentum: B::Float,
    );

    fn input_size(&self) -> usize;
    fn output_size(&self) -> usize;
}
