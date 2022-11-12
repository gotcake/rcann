use std::iter::zip;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use crate::loss::LossFn;
use crate::raw::net::{Layer, Net};
use crate::raw::util::{mat_mul, mat_mul_a_transpose, resize_if_needed, generic_column_sum, sub_assign};

struct TrainBuffers {
    output_error: Vec<f32>,
    first: LayerTrainBuffers,
    hidden: Box<[LayerTrainBuffers]>,
    last: LayerTrainBuffers,
}

struct LayerTrainBuffers {
    activation_errors: Vec<f32>,
    bias_errors: Box<[f32]>,
    weight_errors: Box<[f32]>,
    input_error: Vec<f32>,
    temp: Vec<f32>,
}

impl TrainBuffers {
    fn for_net(net: &Net) -> Self {
        TrainBuffers {
            output_error: Vec::new(),
            first: LayerTrainBuffers::for_layer(&net.first),
            hidden: net.hidden.iter().map(LayerTrainBuffers::for_layer).collect(),
            last: LayerTrainBuffers::for_layer(&net.last),
        }
    }
}

impl LayerTrainBuffers {
    fn for_layer(layer: &Layer) -> Self {
        LayerTrainBuffers {
            activation_errors: Vec::new(),
            bias_errors: vec![0.0; layer.size].into_boxed_slice(),
            weight_errors: vec![0.0; layer.size * layer.prev_size].into_boxed_slice(),
            input_error: Vec::new(),
            temp: Vec::new(),
        }
    }
}

impl Net {

    fn backprop(
        &mut self,
        buffers: &mut TrainBuffers,
        batch_size: usize,
        input: &[f32],
        expected: &[f32],
        loss: &LossFn,
        learn_rate: f32,
        momentum: f32,
    ) {
        assert_eq!(input.len(), self.input_len() * batch_size);
        assert_eq!(expected.len(), self.output_len() * batch_size);

        // propagate error
        {
            resize_if_needed(&mut buffers.output_error, self.output_len() * batch_size);
            loss.derivative_batch_slice(buffers.output_error.as_mut(), &self.last.outputs, expected);

            let last_input = if self.hidden.len() > 0 { self.hidden.last().unwrap().outputs.as_slice() } else { self.first.outputs.as_slice() };
            self.last.backprop(batch_size, &mut buffers.last, last_input, &buffers.output_error, true);

            let mut output_error = buffers.last.input_error.as_slice();
            for ((i, layer), layer_buffers) in zip(self.hidden.iter().enumerate().rev(), buffers.hidden.iter_mut().rev()) {
                let input = if i == 0 { self.first.outputs.as_slice() } else { self.hidden[i].outputs.as_slice() };
                layer.backprop(batch_size, layer_buffers, input, output_error, true);
                output_error = layer_buffers.input_error.as_slice();
            }

            self.first.backprop(batch_size, &mut buffers.first, input, &output_error, false);
        }

        // update weights
        {
            self.first.update(batch_size, &mut buffers.first, input, learn_rate, momentum);

            let mut prev_output = self.first.outputs.as_ref();
            for (layer, layer_buffers) in zip(self.hidden.iter_mut(), buffers.hidden.iter_mut()) {
                layer.update(batch_size, layer_buffers, prev_output, learn_rate, momentum);
                prev_output = layer.outputs.as_ref();
            }

            self.last.update(batch_size, &mut buffers.last, prev_output, learn_rate, momentum);
        }
    }

    pub fn train(
        &mut self,
        inputs: &[f32],
        expected: &[f32],
        loss: &LossFn,
        batch_size: usize,
        learn_rate: f32,
        momentum: f32,
        max_epochs: usize,
        rng: &mut StdRng,
    ) {

        assert!(batch_size > 0);
        assert_eq!(inputs.len() % self.input_len(), 0);
        assert_eq!(expected.len() % self.output_len(), 0);
        assert_eq!(inputs.len() / self.input_len(), expected.len() / self.output_len());

        let mut buffers = TrainBuffers::for_net(&self);
        let input_batch_len = batch_size * self.input_len();
        let expected_batch_len = batch_size * self.output_len();

        let mut batches: Vec<(&[f32], &[f32])> = zip(inputs.chunks(input_batch_len), expected.chunks(expected_batch_len)).collect();

        for epoch in 0..max_epochs {
            let do_print = epoch == max_epochs - 1 || epoch % 10 == 0;
            let mut sum = 0.0;

            batches.shuffle(rng);

            for &(b_input, b_expected) in batches.iter() {

                debug_assert_eq!(b_input.len() % self.input_len(), 0);
                debug_assert_eq!(b_expected.len() % self.output_len(), 0);

                let actual_batch_size = b_input.len() / self.input_len();

                debug_assert_eq!(actual_batch_size, b_expected.len() / self.output_len());
                debug_assert!(actual_batch_size <= batch_size);

                let actual_batch_size = b_input.len() / self.input_len();

                let b_output = self.forward(actual_batch_size, &b_input);
                if do_print {
                    sum += loss.compute_total_slice(actual_batch_size, &b_output, &b_expected)
                }

                self.backprop(
                    &mut buffers,
                    actual_batch_size,
                    &b_input,
                    &b_expected,
                    loss,
                    learn_rate,
                    momentum,
                );

            }
            if do_print {
                println!("Epoch {epoch}: Total MSE={sum}")
            }
        }

    }

}

impl Layer {

    fn backprop(
        &self,
        batch_size: usize,
        buffers: &mut LayerTrainBuffers,
        input: &[f32],
        output_error: &[f32],
        compute_input_error: bool,
    ) {

        let output_buff_len = self.size * batch_size;

        debug_assert_eq!(output_error.len(), output_buff_len);

        resize_if_needed(&mut buffers.activation_errors, output_buff_len);
        self.activation_fn.compute_activation_errors_slice(
            self.prev_size,
            self.size,
            &mut buffers.activation_errors,
            input,
            &self.activations,
            &self.outputs,
            output_error,
            &mut buffers.temp,
        );

        if compute_input_error {
            resize_if_needed(&mut buffers.input_error, self.prev_size * batch_size);
            mat_mul(
                batch_size,
                self.size,
                self.prev_size,
                1.0,
                0.0,
                &buffers.activation_errors,
                &self.weights,
                &mut buffers.input_error,
            );
        }

    }

    fn update(
        &mut self,
        batch_size: usize,
        buffers: &mut LayerTrainBuffers,
        input: &[f32],
        learn_rate: f32,
        momentum: f32,
    ) {

        debug_assert_eq!(input.len(), self.prev_size * batch_size);

        mat_mul_a_transpose(
            self.size,
            batch_size,
            self.prev_size,
            learn_rate,
            momentum,
            &buffers.activation_errors,
            input,
            &mut buffers.weight_errors,
        );

        // compute bias errors
        generic_column_sum(
            batch_size,
            self.size,
            learn_rate,
            momentum,
            &buffers.activation_errors,
            &mut buffers.bias_errors,
        );

        sub_assign(&buffers.weight_errors, &mut self.weights);
        sub_assign(&buffers.bias_errors, &mut self.biases);

    }

}