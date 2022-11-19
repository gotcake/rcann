use crate::backend::Backend;
use crate::loss::LossFn;
use crate::net::initializer::{NetInitializer, RandomNetInitializer};
use crate::net::layer::{ConcreteLayer, ConcreteLayerParams, Layer, LayerParams};
use crate::tensor::{Dims, ITensor};
use std::fmt::{Debug, Formatter};
use std::iter::zip;

pub mod initializer;
pub mod layer;

pub struct Net<B: Backend> {
    backend: B,
    first: ConcreteLayer<B>,
    first_output: B::Tensor,
    hidden: Box<[ConcreteLayer<B>]>,
    hidden_outputs: Box<[B::Tensor]>,
    last: ConcreteLayer<B>,
    last_output: B::Tensor,

    // training only
    hidden_input_error: Box<[B::Tensor]>,
    last_input_error: B::Tensor,
    output_error_buff: B::Tensor,
    output_error_deriv_buff: B::Tensor,
}

impl<B: Backend> Net<B> {
    fn new(
        backend: B,
        first: ConcreteLayer<B>,
        hidden: Box<[ConcreteLayer<B>]>,
        last: ConcreteLayer<B>,
    ) -> Self {
        let first_output = backend.new_tensor((0, first.output_size()));
        let hidden_outputs = hidden
            .iter()
            .map(|l| backend.new_tensor((0, l.output_size())))
            .collect();
        let last_output = backend.new_tensor((0, last.output_size()));
        let hidden_input_error = hidden
            .iter()
            .map(|l| backend.new_tensor((0, l.input_size())))
            .collect();
        let last_input_error = backend.new_tensor((0, last.input_size()));
        let output_error_buff = backend.new_tensor(0);
        let output_error_deriv_buff = backend.new_tensor((0, last.output_size()));
        Net {
            backend,
            first,
            first_output,
            hidden,
            hidden_outputs,
            last,
            last_output,
            output_error_buff,
            output_error_deriv_buff,
            hidden_input_error,
            last_input_error,
        }
    }

    pub fn predict(&mut self, input: &B::Tensor) -> &B::Tensor {
        let num_rows = input.dims().first();
        assert_eq!(
            input.dims(),
            &Dims::D2(num_rows, self.input_size()),
            "Invalid dimensions for input tensor"
        );
        self.forward(num_rows, input);
        &self.last_output
    }

    pub fn train_batch(
        &mut self,
        input: &B::Tensor,
        expected: &B::Tensor,
        loss: &LossFn,
        learn_rate: B::DType,
        momentum: B::DType,
    ) -> TrainBatchResult<B> {
        let num_rows = input.dims().first();
        let input_size = self.input_size();
        let output_size = self.output_size();

        assert_eq!(
            input.dims(),
            &Dims::D2(num_rows, input_size),
            "Invalid dimensions for input tensor"
        );
        assert_eq!(
            expected.dims(),
            &Dims::D2(num_rows, output_size),
            "Invalid dimensions for expected tensor"
        );

        self.forward(num_rows, input);
        self.backprop(num_rows, input, expected, &loss, learn_rate, momentum);

        TrainBatchResult {
            output: &self.last_output,
            error: &self.output_error_buff,
        }
    }

    fn forward(&mut self, num_rows: usize, input: &B::Tensor) {
        self.backend.resize_tensor_first_dim(&mut self.first_output, num_rows);
        self.first
            .forward(&self.backend, input, &mut self.first_output);

        let mut input = &self.first_output;
        for (layer, output) in zip(self.hidden.iter_mut(), self.hidden_outputs.iter_mut()) {
            self.backend.resize_tensor_first_dim(output, num_rows);
            layer.forward(&self.backend, input, output);
            input = output;
        }

        self.backend.resize_tensor_first_dim(&mut self.last_output, num_rows);
        self.last.forward(&self.backend, input, &mut self.last_output);
    }

    fn backprop(
        &mut self,
        num_rows: usize,
        input: &B::Tensor,
        expected: &B::Tensor,
        loss: &LossFn,
        learn_rate: B::DType,
        momentum: B::DType,
    ) {
        self.backend.resize_tensor(&mut self.output_error_buff, num_rows);
        self.backend.resize_tensor_first_dim(&mut self.output_error_deriv_buff, num_rows);
        loss.compute(
            &self.backend,
            &self.last_output,
            expected,
            &mut self.output_error_buff,
            &mut self.output_error_deriv_buff,
        );

        let last_input = match self.hidden_outputs.last() {
            None => &self.first_output,
            Some(last_hidden_output) => last_hidden_output,
        };

        self.backend.resize_tensor_first_dim(&mut self.last_input_error, num_rows);
        self.last.backprop(
            &self.backend,
            last_input,
            &self.last_output,
            Some(&mut self.last_input_error),
            &self.output_error_deriv_buff,
            learn_rate,
            momentum,
        );

        let mut output_error = &self.last_input_error;
        for (i, (layer, (output, input_error))) in zip(
            self.hidden.iter_mut(),
            zip(
                self.hidden_outputs.iter(),
                self.hidden_input_error.iter_mut(),
            ),
        )
        .enumerate()
        .rev()
        {
            let layer_input = if i == 0 {
                &self.first_output
            } else {
                &self.hidden_outputs[i - 1]
            };
            self.backend.resize_tensor_first_dim(input_error, num_rows);
            layer.backprop(
                &self.backend,
                layer_input,
                output,
                Some(input_error),
                output_error,
                learn_rate,
                momentum,
            );
            output_error = input_error;
        }

        self.first.backprop(
            &self.backend,
            input,
            &self.first_output,
            None,
            output_error,
            learn_rate,
            momentum,
        )
    }

    #[inline]
    pub fn input_size(&self) -> usize {
        self.first.input_size()
    }

    #[inline]
    pub fn output_size(&self) -> usize {
        self.last.output_size()
    }
}

pub struct NetBuilder<B: Backend> {
    backend: B,
    input_size: usize,
    initializer: Box<dyn NetInitializer<B::DType>>,
    layers: Vec<ConcreteLayerParams>,
}

impl<B: Backend> NetBuilder<B> {
    pub fn new(backend: B, input_size: usize) -> Self {
        NetBuilder {
            backend,
            input_size,
            initializer: Box::new(RandomNetInitializer::default()),
            layers: Vec::new(),
        }
    }
    pub fn with_initializer<I>(mut self, initializer: I) -> Self
    where
        I: 'static + NetInitializer<B::DType>,
    {
        self.initializer = Box::new(initializer);
        self
    }

    pub fn with_layer<T>(mut self, layer: T) -> Self
    where
        T: Into<ConcreteLayerParams>,
    {
        self.layers.push(layer.into());
        self
    }

    pub fn build(mut self) -> Option<Net<B>> {
        if self.layers.len() < 2 {
            None
        } else {
            let first_params = self.layers.remove(0);
            let last_params = self.layers.pop().unwrap();
            let first = first_params.create_layer(
                &self.backend,
                0,
                self.input_size,
                self.initializer.as_mut(),
            );
            let mut hidden = Vec::with_capacity(self.layers.len());
            let mut last_size = first.output_size();
            let mut layer_idx: usize = 1;
            for layer_param in self.layers.iter() {
                let layer = layer_param.create_layer(
                    &self.backend,
                    layer_idx,
                    last_size,
                    self.initializer.as_mut(),
                );
                last_size = layer.output_size();
                layer_idx += 1;
                hidden.push(layer);
            }
            let last = last_params.create_layer(
                &self.backend,
                layer_idx,
                last_size,
                self.initializer.as_mut(),
            );
            Some(Net::new(
                self.backend,
                first,
                hidden.into_boxed_slice(),
                last,
            ))
        }
    }
}

pub struct TrainBatchResult<'a, B: Backend> {
    pub output: &'a B::Tensor,
    pub error: &'a B::Tensor,
}

impl<B: Backend> Debug for Net<B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Net")
            .field("backend", &self.backend)
            .field("first", &self.first)
            .field("hidden", &self.hidden)
            .field("last", &self.last)
            .finish_non_exhaustive()
        /*write!(f, "Net {{\n   backend: {:?},\n   first: {:?},\n   hidden: [", self.backend, self.first)?;
        for layer in self.hidden.iter() {
            f.write_str("\n      ")?;
            Debug::fmt(layer, f)?;
            f.write_char(',')?;
        }
        write!(f, "\n   ],\n   last: {:?}\n}}", self.last)*/
    }
}
