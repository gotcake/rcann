use crate::backend::Backend;
use crate::dtype::DType;
use crate::loss::LossFn;
use crate::net::initializer::{NetInitializer, RandomNetInitializer};
use crate::net::layer::{ConcreteLayer, ConcreteLayerParams, Layer, LayerParams};
use crate::scoring::{NoOpScorer, Scorer};
use crate::tensor::{Dim0, Dim1, Dim2, ITensor, Tensor1, Tensor2, TensorBase, TensorView2};
use rand::seq::SliceRandom;
use rand::Rng;
use std::fmt::{Debug, Formatter};
use std::iter::zip;

pub mod initializer;
pub mod layer;

struct RawNet<B: Backend> {
    backend: B,
    first: ConcreteLayer<B>,
    first_output: B::Tensor<Dim2>,
    hidden: Box<[ConcreteLayer<B>]>,
    hidden_outputs: Box<[B::Tensor<Dim2>]>,
    last: ConcreteLayer<B>,
    last_output: B::Tensor<Dim2>,

    // training only
    hidden_input_error: Box<[B::Tensor<Dim2>]>,
    last_input_error: B::Tensor<Dim2>,
    output_error_buff: B::Tensor<Dim1>,
    output_error_deriv_buff: B::Tensor<Dim2>,
}

impl<B: Backend> RawNet<B> {
    fn new(backend: B, first: ConcreteLayer<B>, hidden: Box<[ConcreteLayer<B>]>, last: ConcreteLayer<B>) -> Self {
        let first_output = backend.new_tensor_batch_sized(Dim1(first.output_size()));
        let hidden_outputs = hidden
            .iter()
            .map(|l| backend.new_tensor_batch_sized(Dim1(l.output_size())))
            .collect();
        let last_output = backend.new_tensor_batch_sized(Dim1(last.output_size()));
        let hidden_input_error = hidden
            .iter()
            .map(|l| backend.new_tensor_batch_sized(Dim1(l.input_size())))
            .collect();
        let last_input_error = backend.new_tensor_batch_sized(Dim1(last.input_size()));
        let output_error_buff = backend.new_tensor_batch_sized(Dim0);
        let output_error_deriv_buff = backend.new_tensor_batch_sized(Dim1(last.output_size()));
        RawNet {
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

    fn forward(&mut self, input: B::TensorRef<'_, Dim2>) {
        let num_rows = input.dims().rows();
        self.backend.resize_tensor_major(&mut self.first_output, num_rows);
        self.first.forward(&self.backend, input, &mut self.first_output);

        let mut input = &self.first_output;
        for (layer, output) in zip(self.hidden.iter_mut(), self.hidden_outputs.iter_mut()) {
            self.backend.resize_tensor_major(output, num_rows);
            layer.forward(&self.backend, B::TensorRef::from(input), output);
            input = output;
        }

        self.backend.resize_tensor_major(&mut self.last_output, num_rows);
        self.last
            .forward(&self.backend, B::TensorRef::from(input), &mut self.last_output);
    }

    fn backprop(
        &mut self,
        input: B::TensorRef<'_, Dim2>,
        expected: B::TensorRef<'_, Dim2>,
        loss: &LossFn,
        learn_rate: B::Float,
        momentum: B::Float,
    ) {
        let num_rows = input.dims().rows();
        self.backend.resize_tensor(&mut self.output_error_buff, Dim1(num_rows));
        self.backend
            .resize_tensor_major(&mut self.output_error_deriv_buff, num_rows);
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

        self.backend.resize_tensor_major(&mut self.last_input_error, num_rows);
        self.last.backprop(
            &self.backend,
            B::TensorRef::from(last_input),
            &self.last_output,
            Some(&mut self.last_input_error),
            &self.output_error_deriv_buff,
            learn_rate,
            momentum,
        );

        let mut output_error = &self.last_input_error;
        for (i, (layer, (output, input_error))) in zip(
            self.hidden.iter_mut(),
            zip(self.hidden_outputs.iter(), self.hidden_input_error.iter_mut()),
        )
        .enumerate()
        .rev()
        {
            let layer_input = if i == 0 {
                &self.first_output
            } else {
                &self.hidden_outputs[i - 1]
            };
            self.backend.resize_tensor_major(input_error, num_rows);
            layer.backprop(
                &self.backend,
                B::TensorRef::from(layer_input),
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
}

pub struct NetBuilder<B: Backend> {
    backend: B,
    input_size: usize,
    initializer: Box<dyn NetInitializer<B::Float>>,
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
        I: 'static + NetInitializer<B::Float>,
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
            let first = first_params.create_layer(&self.backend, 0, self.input_size, self.initializer.as_mut());
            let mut hidden = Vec::with_capacity(self.layers.len());
            let mut last_size = first.output_size();
            let mut layer_idx: usize = 1;
            for layer_param in self.layers.iter() {
                let layer = layer_param.create_layer(&self.backend, layer_idx, last_size, self.initializer.as_mut());
                last_size = layer.output_size();
                layer_idx += 1;
                hidden.push(layer);
            }
            let last = last_params.create_layer(&self.backend, layer_idx, last_size, self.initializer.as_mut());
            Some(Net::new(RawNet::new(
                self.backend,
                first,
                hidden.into_boxed_slice(),
                last,
            )))
        }
    }
}

pub struct RawTrainBatchResult<'a, B: Backend> {
    pub output: &'a B::Tensor<Dim2>,
    pub error: &'a B::Tensor<Dim1>,
}

pub struct TrainBatchResult<'a, T> {
    pub output: &'a Tensor2<T>,
    pub error: &'a Tensor1<T>,
}

pub struct Net<B: Backend> {
    raw: RawNet<B>,
    input_buff: B::InputAdaptionBuff<Dim2>,
    output_buff: B::OutputAdaptionBuff<Dim2>,
    // training only
    expected_buff: B::InputAdaptionBuff<Dim2>,
    error_buff: B::OutputAdaptionBuff<Dim1>,
}

impl<B: Backend> Net<B> {
    fn new(raw: RawNet<B>) -> Self {
        let input_size = raw.first.input_size();
        let output_size = raw.last.output_size();
        let input_buff = raw.backend.new_input_adaption_buff(Dim1(input_size));
        let output_buff = raw.backend.new_output_adaption_buff(Dim1(output_size));
        let expected_buff = raw.backend.new_input_adaption_buff(Dim1(output_size));
        let error_buff = raw.backend.new_output_adaption_buff(Dim0);
        Self {
            raw,
            input_buff,
            output_buff,
            expected_buff,
            error_buff,
        }
    }

    pub fn predict(&mut self, input: TensorView2<B::Float>) -> &Tensor2<B::Float> {
        let &Dim2(num_rows, num_cols) = input.dims();
        let max_batch_size = self.max_batch_size();
        assert_eq!(
            num_cols,
            self.input_size(),
            "Invalid number of columns for input tensor"
        );
        assert!(
            num_rows <= max_batch_size,
            "Invalid number of rows for input tensor: {num_rows}. Max allowed: {max_batch_size}.",
        );
        let input = self.raw.backend.adapt_input(&mut self.input_buff, input);
        self.raw.forward(input);
        self.raw
            .backend
            .adapt_output(&mut self.output_buff, &self.raw.last_output)
    }

    pub fn train_batch(
        &mut self,
        input: TensorView2<B::Float>,
        expected: TensorView2<B::Float>,
        loss: &LossFn,
        learn_rate: B::Float,
        momentum: B::Float,
    ) -> TrainBatchResult<B::Float> {
        let &Dim2(num_rows, num_cols) = input.dims();
        let max_batch_size = self.max_batch_size();

        assert_eq!(
            num_cols,
            self.input_size(),
            "Invalid number of columns for input tensor"
        );
        assert!(
            num_rows <= max_batch_size,
            "Invalid number of rows for input tensor: {num_rows}. Max allowed: {max_batch_size}.",
        );
        assert_eq!(
            expected.dims(),
            &Dim2(num_rows, self.output_size()),
            "Invalid dimensions for expected tensor"
        );

        let input = self.raw.backend.adapt_input(&mut self.input_buff, input);
        let expected = self.raw.backend.adapt_input(&mut self.expected_buff, expected);

        self.raw.forward(input.clone());
        self.raw.backprop(input, expected, &loss, learn_rate, momentum);

        let output = self
            .raw
            .backend
            .adapt_output(&mut self.output_buff, &self.raw.last_output);
        let error = self
            .raw
            .backend
            .adapt_output(&mut self.error_buff, &self.raw.output_error_buff);

        TrainBatchResult { output, error }
    }

    fn train_epoch<S: Scorer<B>>(
        &mut self,
        batches: &[(B::TensorRef<'_, Dim2>, B::TensorRef<'_, Dim2>)],
        loss: &LossFn,
        learn_rate: B::Float,
        momentum: B::Float,
        scorer: &mut S,
    ) {
        for (input, expected) in batches {
            let num_rows = input.dims().rows();
            debug_assert_eq!(num_rows, expected.dims().rows());
            debug_assert_eq!(input.dims().cols(), self.input_size());
            debug_assert_eq!(expected.dims().cols(), self.output_size());
            self.raw.forward(input.clone());
            self.raw
                .backprop(input.clone(), expected.clone(), &loss, learn_rate, momentum);
            scorer.process_batch(&self.raw.backend, &self.raw.last_output, expected.clone());
            self.raw.backend.flush();
        }
        self.raw.backend.sync();
    }

    pub fn train<R: Rng>(
        &mut self,
        rng: &mut R,
        input: TensorView2<B::Float>,
        expected: TensorView2<B::Float>,
        num_epochs: usize,
    ) {
        let input_size = self.input_size();
        let output_size = self.output_size();
        assert_eq!(
            input.dims().rows(),
            expected.dims().rows(),
            "Mismatched number of rows in input and expected"
        );
        assert_eq!(
            input.dims().cols(),
            self.input_size(),
            "Mismatched number of columns in input"
        );
        assert_eq!(
            expected.dims().cols(),
            self.output_size(),
            "Mismatched number of columns in expected"
        );
        let batch_size = self.raw.backend.max_batch_size();
        let num_batches = (input.dims().rows() as f64 / batch_size as f64).ceil() as usize;

        // allocate buffers for the input and expected data
        let mut buffers: Vec<_> = (0..num_batches)
            .map(|_| {
                (
                    self.raw.backend.new_input_adaption_buff(Dim1(input_size)),
                    self.raw.backend.new_input_adaption_buff(Dim1(output_size)),
                )
            })
            .collect();

        let mut batches = Vec::with_capacity(num_batches);

        // convert input to backend format
        for ((input_batch, expected_batch), (input_buff, expected_buff)) in zip(
            zip(
                input.iter_major_axis_chunks(batch_size),
                expected.iter_major_axis_chunks(batch_size),
            ),
            buffers.iter_mut(),
        ) {
            batches.push((
                self.raw.backend.adapt_input(input_buff, input_batch),
                self.raw.backend.adapt_input(expected_buff, expected_batch),
            ));
        }

        for i in 0..num_epochs {
            batches.shuffle(rng);
            self.train_epoch(
                &batches,
                &LossFn::MSE,
                B::Float::from_f64(0.1),
                B::Float::from_f64(0.1),
                &mut NoOpScorer,
            );
            println!("epoch {i}");
        }
    }

    pub fn evaluate<S: Scorer<B>>(
        &mut self,
        input: TensorView2<B::Float>,
        expected: TensorView2<B::Float>,
        scorer: &mut S,
    ) {
        assert_eq!(
            input.dims().rows(),
            expected.dims().rows(),
            "Mismatched number of rows in inputs and expected"
        );
        assert_eq!(
            input.dims().cols(),
            self.input_size(),
            "Mismatched number of columns in inputs"
        );
        assert_eq!(
            expected.dims().cols(),
            self.output_size(),
            "Mismatched number of columns in expected"
        );
        let batch_size = self.raw.backend.max_batch_size();
        for (input_batch, expected_batch) in zip(
            input.iter_major_axis_chunks(batch_size),
            expected.iter_major_axis_chunks(batch_size),
        ) {
            let input_batch = self.raw.backend.adapt_input(&mut self.input_buff, input_batch);
            let expected_batch = self.raw.backend.adapt_input(&mut self.expected_buff, expected_batch);
            self.raw.forward(input_batch);
            scorer.process_batch(&self.raw.backend, &self.raw.last_output, expected_batch)
        }
    }

    #[inline]
    pub fn input_size(&self) -> usize {
        self.raw.first.input_size()
    }

    #[inline]
    pub fn output_size(&self) -> usize {
        self.raw.last.output_size()
    }

    #[inline]
    pub fn backend(&self) -> &B {
        &self.raw.backend
    }

    #[inline]
    pub fn max_batch_size(&self) -> usize {
        self.raw.backend.max_batch_size()
    }
}

impl<B: Backend> Debug for Net<B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Net")
            .field("backend", &self.raw.backend)
            .field("first", &self.raw.first)
            .field("hidden", &self.raw.hidden)
            .field("last", &self.raw.last)
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
