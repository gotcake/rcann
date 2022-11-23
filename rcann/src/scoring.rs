use std::fmt::Debug;
use crate::backend::Backend;
use crate::net::Net;
use crate::tensor::Dim2;

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
}

impl<B: Backend> MulticlassScorer<B> where B::Float: Debug {
    fn new(backend: &B, output_size: usize) -> Self {
        MulticlassScorer {
            matrix: backend.new_tensor_exact(Dim2(output_size, output_size))
        }
    }
    pub fn for_net(net: &Net<B>) -> Self {
        Self::new(net.backend(), net.output_size())
    }
    pub fn print_report(&self, backend: &B) {
        let matrix = backend.tensor_as_native(&self.matrix);
        println!("Confusion Matrix: {matrix:?}");
    }
}

impl<B: Backend> Scorer<B> for MulticlassScorer<B> {
    fn process_batch(&mut self, backend: &B, output: &B::Tensor<Dim2>, expected: B::TensorRef<'_, Dim2>) {
        backend.accum_confusion_matrix_multiclass(&mut self.matrix, output, expected);
    }
}