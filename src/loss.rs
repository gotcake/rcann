use std::iter::zip;

#[derive(Copy, Clone)]
pub enum LossFn {
    MSE,
}

impl LossFn {
    pub fn compute_total_slice(&self, batch_size: usize, output: &[f32], expected: &[f32]) -> f32 {
        debug_assert_eq!(output.len(), expected.len());
        debug_assert_eq!(output.len() % batch_size, 0);
        match self {
            LossFn::MSE => {
                let cols = output.len() / batch_size;
                zip(output.chunks_exact(cols), expected.chunks_exact(cols))
                    .fold(0.0, |sum, (o_row, e_row)| {
                        sum + zip(o_row, e_row).fold(0.0, |acc, (&o, &e)| {
                            let diff = o - e;
                            acc + diff * diff
                        }) / cols as f32
                    })
            }
        }
    }
    pub fn derivative_batch_slice(&self, target: &mut [f32], output: &[f32], expected: &[f32]) {
        debug_assert_eq!(target.len(), output.len());
        debug_assert_eq!(output.len(), expected.len());
        match self {
            LossFn::MSE  => {
                for (t, (&o, &e)) in zip(target, zip(output, expected)) {
                    *t = o - e;
                }
            }
        }
    }
}
