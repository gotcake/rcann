mod fully_connected;
mod concrete;

use std::fmt::Debug;
use crate::backend::Backend;
use crate::net::initializer::NetInitializer;

pub use fully_connected::{FullyConnectedLayerParams, FullyConnectedLayer};
pub use concrete::{ConcreteLayerParams, ConcreteLayer};

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
        initializer: &mut dyn NetInitializer<B::DType>
    ) -> Self::Layer;

}

pub trait Layer<B: Backend>: Debug {

    fn forward(
        &mut self,
        backend: &B,
        input: &B::Tensor,
        output: &mut B::Tensor,
    );

    fn backprop(
        &mut self,
        backend: &B,
        input: &B::Tensor,
        output: &B::Tensor,
        input_error: Option<&mut B::Tensor>,
        output_error: &B::Tensor,
        learn_rate: B::DType,
        momentum: B::DType,
    );

    fn input_size(&self) -> usize;
    fn output_size(&self) -> usize;

}



