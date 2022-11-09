use num_traits::{One, Zero};
use crate::activation::ActivationFn;
use crate::backend::Backend;
use crate::tensor::{Tensor2};

pub trait Layer<B: Backend> {

    fn forward(
        &mut self,
        backend: &B,
        input: &B::Tensor2,
        output: &mut B::Tensor2,
    );

    fn backprop(
        &mut self,
        backend: &B,
        input: &B::Tensor2,
        output: &B::Tensor2,
        output_error: &B::Tensor2,
        input_error: Option<&mut B::Tensor2>,
        learn_rate: B::DType,
        momentum: B::DType,
    );

    fn input_size(&self) -> usize;
    fn output_size(&self) -> usize;

}

struct FullyConnectedLayer<B: Backend> {
    prev_size: usize,
    size: usize,
    weights: B::Tensor2,
    biases: B::Tensor,
    activation: B::Tensor2,
    training_tensors: Option<TrainingTensors<B>>,
    activation_fn: ActivationFn,
}

struct TrainingTensors<B: Backend> {
    activation_error: B::Tensor2,
    weight_error: B::Tensor2,
    bias_error: B::Tensor,
}

impl<B: Backend> TrainingTensors<B> {
    fn new(backend: &B, num_rows: usize, size: usize, prev_size: usize) -> Self {
        TrainingTensors {
            activation_error: backend.new_tensor2(num_rows, size),
            weight_error: backend.new_tensor2(size, prev_size),
            bias_error: backend.new_tensor1(size),
        }
    }

}

impl<B: Backend> FullyConnectedLayer<B> {
    fn new(backend: &B, prev_size: usize, size: usize, activation_fn: ActivationFn) -> Self {
        FullyConnectedLayer {
            prev_size,
            size,
            weights: backend.new_tensor2(size, prev_size),
            biases: backend.new_tensor1(size),
            activation: backend.new_tensor2(0, size),
            training_tensors: None,
            activation_fn,
        }
    }
}

impl<B: Backend> Layer<B> for FullyConnectedLayer<B> {
    fn forward(&mut self, backend: &B, input: &B::Tensor2, output: &mut B::Tensor2) {
        let num_rows = input.rows();

        debug_assert_eq!(input.rows(), output.rows());
        debug_assert_eq!(input.cols(), self.prev_size);
        debug_assert_eq!(output.cols(), self.size);

        backend.resize_tensor2(&mut self.activation, num_rows, self.size);
        backend.matmul(
            B::DType::one(),
            input,
            false,
            &self.weights,
            true,
            B::DType::zero(),
            &mut self.activation,
            false
        );

        self.activation_fn.compute(
            backend,
            &self.activation,
            output
        );

    }

    fn backprop(
        &mut self,
        backend: &B,
        input: &B::Tensor2,
        output: &B::Tensor2,
        out_error: &B::Tensor2,
        input_error: Option<&mut B::Tensor2>,
        learn_rate: B::DType,
        momentum: B::DType,
    ) {

        let num_rows = input.rows();

        debug_assert_eq!(input.cols(), self.prev_size);
        debug_assert_eq!(output.cols(), self.size);
        debug_assert_eq!(input.rows(), output.rows());
        debug_assert_eq!(output.dim(), out_error.dim());

        let tt = self.training_tensors.get_or_insert_with(|| {
            TrainingTensors::new(backend, input.rows(), self.size, self.prev_size)
        });

        backend.resize_tensor2(&mut tt.activation_error, num_rows, self.size);
        self.activation_fn.compute_error(
            backend,
            &self.activation,
            output,
            out_error,
            &mut tt.activation_error,
        );

        if let Some(input_error) = input_error {
            debug_assert_eq!(input.dim(), input_error.dim());
            backend.matmul(
                B::DType::one(),
                &tt.activation_error,
                false,
                &self.weights,
                false,
                B::DType::zero(),
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

        backend.add_assign2(
            -B::DType::one(),
            &tt.weight_error,
            B::DType::one(),
            &mut self.weights
        );

        backend.add_assign(
            -B::DType::one(),
            &tt.bias_error,
            B::DType::one(),
            &mut self.biases
        );

    }

    #[inline]
    fn input_size(&self) -> usize {
        self.prev_size
    }

    #[inline]
    fn output_size(&self) -> usize {
        self.size
    }
}