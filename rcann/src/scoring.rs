use crate::backend::Backend;
use crate::net::Net;
use crate::tensor::Dim2;

pub trait Scorer<B: Backend> {
    fn process_batch(&mut self, backend: &B, output: &B::Tensor<Dim2>, expected: &B::Tensor<Dim2>);
}

pub struct NoOpScorer;

impl<B: Backend> Scorer<B> for NoOpScorer {
    #[inline]
    fn process_batch(&mut self, _backend: &B, _output: &B::Tensor<Dim2>, _expected: &B::Tensor<Dim2>) {}
}

pub struct MulticlassScorer<B: Backend> {
    matrix: B::Tensor<Dim2>,
}

impl<B: Backend> MulticlassScorer<B> {
    fn new(backend: &B, output_size: usize) -> Self {
        MulticlassScorer {
            matrix: backend.new_tensor_exact(Dim2(output_size, output_size))
        }
    }
    pub fn for_net(net: &Net<B>) -> Self {
        Self::new(net.backend(), net.output_size())
    }
}

impl<B: Backend> Scorer<B> for MulticlassScorer<B> {
    fn process_batch(&mut self, backend: &B, output: &B::Tensor<Dim2>, expected: &B::Tensor<Dim2>) {
        backend.accum_confusion_matrix_multiclass(output, expected, &mut self.matrix);
    }
}