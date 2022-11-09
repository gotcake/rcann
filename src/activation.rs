use std::cmp::Ordering;
use std::iter::{zip};
use serde::{Deserialize, Serialize};
use crate::backend::Backend;
use crate::dtype::DType;
use crate::raw::util::{mat_mul, resize_if_needed};

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum ActivationFn {
    Sigmoid,
    ReLU { leak: f64 },
    Softmax,
}

impl ActivationFn {
    pub fn compute_slice(&self, size: usize, target: &mut [f32], activations: &[f32]) {
        debug_assert_eq!(target.len(), activations.len());
        debug_assert_eq!(target.len() % size, 0);
        match self {
            ActivationFn::Sigmoid => {
                for (t, &x) in zip(target, activations) {
                    *t = 1.0 / (1.0 + (-x).exp());
                }
            }
            &ActivationFn::ReLU { leak } => {
                let slope = leak as f32;
                for (t, &x) in zip(target, activations) {
                    *t = if x < 0.0 { x * slope } else { x }
                }
            }
            ActivationFn::Softmax => {
                for (t_arr, a_arr) in zip(target.chunks_exact_mut(size), activations.chunks_exact(size)) {
                    // shift the values by -max(inputs) to prevent overflow (does not affect derivative)
                    let max = a_arr.iter().max_by(|a, b| if a > b { Ordering::Greater } else { Ordering::Less }).unwrap();
                    let mut sum = 0.0;
                    for (t, &a) in zip(t_arr.iter_mut(), a_arr) {
                        let x = (a - max).exp();
                        sum += x;
                        *t = x;
                    }
                    for t in t_arr.iter_mut() {
                        *t /= sum
                    }
                }
            }
        }
    }

    pub fn compute<B: Backend>(&self, backend: &B, activation: &B::Tensor2, output: &mut B::Tensor2) {
        match self {
            ActivationFn::Sigmoid => { backend.sigmoid(activation, output) }
            &ActivationFn::ReLU { leak } => { backend.relu(B::DType::from_f64(leak), activation, output) }
            ActivationFn::Softmax => { backend.softmax(activation, output) }
        }
    }

    pub fn compute_error<B: Backend>(&self, backend: &B, activation: &B::Tensor2, output: &B::Tensor2, out_error: &B::Tensor2, result: &mut B::Tensor2) {
        match self {
            ActivationFn::Sigmoid => { backend.sigmoid_error(output , out_error, result) }
            &ActivationFn::ReLU { leak } => { backend.relu_error(B::DType::from_f64(leak), activation, out_error, result) }
            ActivationFn::Softmax => { backend.softmax_error(output, out_error, result) }
        }
    }

    pub fn compute_activation_errors_slice(
        &self,
        prev_size: usize,
        size: usize,
        target: &mut [f32],
        input: &[f32],
        activations: &[f32],
        outputs: &[f32],
        output_error: &[f32],
        temp: &mut Vec<f32>,
    ) {
        debug_assert_eq!(target.len(), activations.len());
        debug_assert_eq!(outputs.len(), output_error.len());
        debug_assert_eq!(target.len(), output_error.len());
        debug_assert_eq!(outputs.len() % size, 0);
        debug_assert_eq!(input.len() % prev_size, 0);
        debug_assert_eq!(input.len() / prev_size, outputs.len() / size);

        match self {
            ActivationFn::Sigmoid => {
                for ((t, &out), &out_err) in zip(zip(target, outputs), output_error) {
                    *t = out_err * (out * (1.0 - out))
                }
            }
            &ActivationFn::ReLU { leak } => {
                let slope = leak as f32;
                for ((t, &act), &out_err) in zip(zip(target, activations), output_error) {
                    *t = if act < 0.0 { slope * out_err } else { out_err };
                }
            }
            /*
            def l_dir_shortcut(W, S, x):
                dir_matrix = np.zeros((W.shape[0] * W.shape[1], W.shape[1]))

                for t in range(0, W.shape[1]):
                    for i in range(0, W.shape[1]):
                        for j in range(0, W.shape[0]):
                            dir_matrix[(i*W.shape[0]) + j][t] = S[t] * ((i==t) - S[i]) * x[j]

                return dir_matrix
             */
            ActivationFn::Softmax => {
                let size2 = size * size;
                resize_if_needed(temp, size2);
                //let (temp1, temp2) = temp.split_at_mut(size2);
                for (t_arr, (o_arr, oerr_arr)) in zip(
                    target.chunks_exact_mut(size),
                    zip(outputs.chunks_exact(size), output_error.chunks_exact(size))
                ) {

                    for i in 0..size {
                        for j in 0..size {
                            if i == j {
                                temp[i * size + j] = o_arr[i] * (1.0 - o_arr[i]);
                            } else {
                                temp[i * size + j] = - o_arr[i] * o_arr[j];
                            }
                        }
                    }

                    mat_mul(
                        1, size, size,
                        1.0,
                        0.0,
                        oerr_arr,
                        temp.as_mut_slice(),
                        t_arr
                    );

                }
            }
        }
    }


}