use crate::backend::Backend;
use crate::dtype::DType;
use crate::net::Net;
use crate::tensor::{Dim2, ITensor, TensorBase, TensorBaseMut};
use std::fmt::Debug;

pub trait Scorer<B: Backend> {
    fn process_batch(&mut self, backend: &B, output: &B::Tensor<Dim2>, expected: B::TensorRef<'_, Dim2>);
}

pub struct NoOpScorer;

impl<B: Backend> Scorer<B> for NoOpScorer {
    #[inline]
    fn process_batch(&mut self, _backend: &B, _output: &B::Tensor<Dim2>, _expected: B::TensorRef<'_, Dim2>) {}
}

pub struct MulticlassScorer<B: Backend> {
    matrix: B::Tensor<Dim2>,
    count: usize,
}

impl<B: Backend> MulticlassScorer<B>
where
    B::Float: Debug,
{
    fn new(backend: &B, output_size: usize) -> Self {
        let matrix = backend.new_tensor_exact(Dim2(output_size, output_size));
        MulticlassScorer { matrix, count: 0 }
    }
    pub fn for_net(net: &Net<B>) -> Self {
        Self::new(net.backend(), net.output_size())
    }
    pub fn print_report(&self, backend: &B) {
        let mut matrix = backend.tensor_as_native(&self.matrix);
        let count = self.count;
        let mut total_correct = 0;
        for (i, mut row) in matrix.iter_major_axis_mut().enumerate() {
            total_correct += row[i].to_usize();
            let total = B::Float::from_usize(row.iter().map(|e| e.to_usize()).sum());
            row.iter_mut().for_each(|e| *e /= total);
        }
        let total_incorrect = count - total_correct;
        let percent_incorrect = (total_incorrect as f64 / self.count as f64) * 100.0;
        println!("Confusion Matrix: {matrix:.3?}");
        println!("Error rate: {percent_incorrect:.2}% ({total_incorrect}/{count})");
    }
}

impl<B: Backend> Scorer<B> for MulticlassScorer<B> {
    fn process_batch(&mut self, backend: &B, output: &B::Tensor<Dim2>, expected: B::TensorRef<'_, Dim2>) {
        self.count += output.dims().rows();
        backend.accum_confusion_matrix_multiclass(&mut self.matrix, output, expected);
    }
}
