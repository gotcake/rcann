use std::iter::zip;
use num_traits::{One, ToPrimitive, Zero};
use crate::backend::Backend;
use crate::dtype::DType;
use crate::tensor::{Dims, ITensor, ITensorBase};
use crate::util::max_index;

pub trait Transform<B: Backend> {

    fn forward(
        &mut self,
        backend: &B,
        input: &B::Tensor,
    ) -> &B::Tensor;

    fn reverse(
        &mut self,
        backend: &B,
        output: &B::Tensor,
    ) -> &B::Tensor;

    fn input_size(&self) -> usize;

    fn output_size(&self) -> usize;

}

pub struct OneHotArgMaxTransform<B: Backend> {
    size: usize,
    output: B::Tensor,
    input: B::Tensor,
}

impl<B: Backend> OneHotArgMaxTransform<B> {
    pub fn new(backend: B, size: usize) -> Self {
        OneHotArgMaxTransform {
            size,
            input: backend.new_tensor((0, size)),
            output: backend.new_tensor((0, 1))
        }
    }
}

impl<B: Backend> Transform<B> for OneHotArgMaxTransform<B> {

    fn forward(&mut self, backend: &B, input: &B::Tensor) -> &B::Tensor {
        let num_rows = input.dims().first();
        assert_eq!(input.dims(), &Dims::D2(num_rows, self.size));
        self.output.resize((num_rows, 1));
        backend.transform_func(input, &mut self.output, |inp_view, mut out_view| {
           for (&inp, out) in zip(inp_view, out_view.chunks_exact_mut(self.size)) {
               out.fill(B::DType::zero());
               if let Some(i) = inp.to_usize() {
                   if i < self.size {
                       out[i] = B::DType::one();
                   }
               }
           }
        });
        &self.output
    }

    fn reverse(&mut self, backend: &B, output: &B::Tensor) -> &B::Tensor {
        let num_rows = output.dims().first();
        assert_eq!(output.dims(), &Dims::D2(num_rows, 1));
        self.input.resize((num_rows, self.size));
        backend.transform_func(output, &mut self.input, |out_view, inp_view| {
            for (inp, out) in zip(inp_view, out_view.chunks_exact(self.size)) {
                *inp = B::DType::from_usize(max_index(out));
            }
        });
        &self.input
    }

    fn input_size(&self) -> usize {
        self.size
    }

    fn output_size(&self) -> usize {
        1
    }

}