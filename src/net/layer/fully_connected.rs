use std::fmt::{Debug, Formatter};
use crate::activation::ActivationFn;
use crate::backend::Backend;
use crate::dtype::DType;
use crate::net::layer::{NetInitializer, Layer, LayerParams, LayerType, ConcreteLayerParams};
use crate::tensor::{Dims, ITensor, ITensorBase};

#[derive(Clone, Debug, PartialEq)]
pub struct FullyConnectedLayerParams {
    pub size: usize,
    pub activation_fn: ActivationFn
}

impl<B: Backend> LayerParams<B> for FullyConnectedLayerParams {

    type Layer = FullyConnectedLayer<B>;

    fn create_layer(
        &self,
        backend: &B,
        layer_idx: usize,
        input_size: usize,
        initializer: &mut dyn NetInitializer<B::DType>
    ) -> Self::Layer where {
        let output_size = self.size;
        let weights = initializer.get_weights(
            LayerType::FullyConnected,
            Dims::D2(output_size, input_size),
            layer_idx,
            input_size,
            output_size
        );
        let biases = initializer.get_biases(
            LayerType::FullyConnected,
            Dims::D1(output_size),
            layer_idx,
        );
        FullyConnectedLayer {
            input_size,
            output_size,
            weights: backend.new_tensor_from_native(weights),
            biases: backend.new_tensor_from_native(biases),
            activation: backend.new_tensor((0, output_size)),
            training_tensors: None,
            activation_fn: self.activation_fn,
        }
    }
}

impl Into<ConcreteLayerParams> for FullyConnectedLayerParams {
    fn into(self) -> ConcreteLayerParams {
        ConcreteLayerParams::FullyConnected(self)
    }
}

pub struct FullyConnectedLayer<B: Backend> {
    input_size: usize,
    output_size: usize,
    weights: B::Tensor,
    biases: B::Tensor,
    activation: B::Tensor,
    training_tensors: Option<TrainingTensors<B>>,
    activation_fn: ActivationFn,
}

impl<B: Backend> FullyConnectedLayer<B> {
    pub fn new(
        backend: &B,
        input_size: usize,
        output_size: usize,
        activation_fn: ActivationFn,

    ) -> Self {
        FullyConnectedLayer {
            input_size,
            output_size,
            weights: backend.new_tensor((output_size, input_size)),
            biases: backend.new_tensor(output_size),
            activation: backend.new_tensor((0, output_size)),
            training_tensors: None,
            activation_fn,
        }
    }
}

struct TrainingTensors<B: Backend> {
    activation_error: B::Tensor,
    weight_error: B::Tensor,
    bias_error: B::Tensor,
}

impl<B: Backend> TrainingTensors<B> {
    fn new(backend: &B, num_rows: usize, size: usize, prev_size: usize) -> Self {
        TrainingTensors {
            activation_error: backend.new_tensor((num_rows, size)),
            weight_error: backend.new_tensor((size, prev_size)),
            bias_error: backend.new_tensor(size),
        }
    }

}

impl<B: Backend> Layer<B> for FullyConnectedLayer<B> {

    fn forward(&mut self, backend: &B, input: &B::Tensor, output: &mut B::Tensor) {
        let num_rows = input.dims().first();

        assert_eq!(input.dims(), &Dims::D2(num_rows, self.input_size), "Invalid dimensions for input tensor");
        assert_eq!(output.dims(), &Dims::D2(num_rows, self.output_size), "Invalid dimensions for output tensor");

        self.activation.resize_first_dim(num_rows);
        backend.matmul(
            B::DType::ONE,
            input,
            false,
            &self.weights,
            true,
            B::DType::ZERO,
            &mut self.activation,
            false
        );

        self.activation_fn.compute(
            backend,
            &self.activation,
            output,
        );
    }

    fn backprop(
        &mut self,
        backend: &B,
        input: &B::Tensor,
        output: &B::Tensor,
        input_error: Option<&mut B::Tensor>,
        out_error: &B::Tensor,
        learn_rate: B::DType,
        momentum: B::DType,
    ) {

        let num_rows = input.dims().first();

        assert_eq!(input.dims(), &Dims::D2(num_rows, self.input_size), "Invalid dimensions for input tensor");
        assert_eq!(output.dims(), &Dims::D2(num_rows, self.output_size), "Invalid dimensions for output tensor");
        assert_eq!(out_error.dims(), &Dims::D2(num_rows, self.output_size), "Invalid dimensions for out_error tensor");

        let tt = self.training_tensors.get_or_insert_with(|| {
            TrainingTensors::new(backend, num_rows, self.output_size, self.input_size)
        });

        tt.activation_error.resize_first_dim(num_rows);
        self.activation_fn.compute_error(
            backend,
            &self.activation,
            output,
            out_error,
            &mut tt.activation_error,
        );

        if let Some(input_error) = input_error {
            assert_eq!(input_error.dims(), &Dims::D2(num_rows, self.input_size));
            backend.matmul(
                B::DType::ONE,
                &tt.activation_error,
                false,
                &self.weights,
                false,
                B::DType::ZERO,
                input_error,
                false
            );
        }

        backend.matmul(
            learn_rate,
            &tt.activation_error,
            true,
            input,
            false,
            momentum,
            &mut tt.weight_error,
            false
        );

        backend.column_sum(
            learn_rate,
            &tt.activation_error,
            momentum,
            &mut tt.bias_error
        );

        backend.add_assign(
            -B::DType::ONE,
            &tt.weight_error,
            B::DType::ONE,
            &mut self.weights
        );

        backend.add_assign(
            -B::DType::ONE,
            &tt.bias_error,
            B::DType::ONE,
            &mut self.biases
        );
    }

    #[inline]
    fn input_size(&self) -> usize {
        self.input_size
    }

    #[inline]
    fn output_size(&self) -> usize {
        self.output_size
    }

}

impl<B: Backend> Debug for FullyConnectedLayer<B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FullyConnectedLayer")
            .field("size", &self.output_size)
            .field("activation_fn", &self.activation_fn)
            .field("weights", &self.weights)
            .field("biases", &self.biases)
            .finish_non_exhaustive()
    }
}