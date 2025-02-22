use super::{DenseLayer, DenseLayerParams, Layer, LayerParams};
use crate::backend::Backend;
use crate::net::initializer::NetInitializer;
use crate::tensor::Dim2;
use std::fmt::{Debug, Formatter};

// This is needed because we can't put dyn LayerParam in a Box :(
// Much sad :(
// Probably because GATs aren't object-safe, but I'm not certain.

#[derive(Clone, Debug, PartialEq)]
pub enum ConcreteLayerParams {
    FullyConnected(DenseLayerParams),
}

impl<B: Backend> LayerParams<B> for ConcreteLayerParams {
    type Layer = ConcreteLayer<B>;
    fn create_layer(
        &self,
        backend: &B,
        layer_idx: usize,
        input_size: usize,
        initializer: &mut dyn NetInitializer<B::Float>,
    ) -> Self::Layer {
        match self {
            ConcreteLayerParams::FullyConnected(params) => {
                ConcreteLayer::FullyConnected(params.create_layer(backend, layer_idx, input_size, initializer))
            }
        }
    }
}

pub enum ConcreteLayer<B: Backend> {
    FullyConnected(DenseLayer<B>),
}

impl<B: Backend> ConcreteLayer<B> {
    fn inner(&self) -> &dyn Layer<B> {
        match self {
            ConcreteLayer::FullyConnected(inner) => inner,
        }
    }
    fn inner_mut(&mut self) -> &mut dyn Layer<B> {
        match self {
            ConcreteLayer::FullyConnected(inner) => inner,
        }
    }
}

impl<B: Backend> Layer<B> for ConcreteLayer<B> {
    fn forward(&mut self, backend: &B, input: B::TensorRef<'_, Dim2>, output: &mut B::Tensor<Dim2>) {
        self.inner_mut().forward(backend, input, output)
    }

    fn backprop(
        &mut self,
        backend: &B,
        input: B::TensorRef<'_, Dim2>,
        output: &B::Tensor<Dim2>,
        input_error: Option<&mut B::Tensor<Dim2>>,
        output_error: &B::Tensor<Dim2>,
        learn_rate: B::Float,
        momentum: B::Float,
    ) {
        self.inner_mut()
            .backprop(backend, input, output, input_error, output_error, learn_rate, momentum)
    }

    #[inline]
    fn input_size(&self) -> usize {
        self.inner().input_size()
    }

    #[inline]
    fn output_size(&self) -> usize {
        self.inner().output_size()
    }
}

impl<B: Backend> Debug for ConcreteLayer<B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self.inner(), f)
    }
}
