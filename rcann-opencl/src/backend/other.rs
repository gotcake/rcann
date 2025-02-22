use crate::backend::OpenCLBackend;
use crate::kernels::softmax::Softmax;
use crate::wrap_cl_error;
use rcann::backend::BackendOther;
use rcann::tensor::{Dim1, Dim2, Dims, ITensor};
use crate::kernels::mse::MSEProgram;
use crate::tensor::OclFloat;

#[allow(unused)]
impl<F: OclFloat> BackendOther for OpenCLBackend<F> {
    fn column_sum(&self, alpha: Self::Float, a: &Self::Tensor<Dim2>, beta: Self::Float, b: &mut Self::Tensor<Dim1>) {
        self.general_program.column_sum(&self.queue, alpha, a, beta, b);
    }

    fn add_assign<D>(&self, alpha: Self::Float, a: &Self::Tensor<D>, beta: Self::Float, b: &mut Self::Tensor<D>)
    where
        D: Dims,
    {
        self.general_program.add_assign(&self.queue, alpha, a, beta, b);
    }

    fn sigmoid(&self, activation: &Self::Tensor<Dim2>, output: &mut Self::Tensor<Dim2>) {
        self.general_program.sigmoid(&self.queue, activation, output);
    }

    fn sigmoid_error(
        &self,
        output: &Self::Tensor<Dim2>,
        out_error: &Self::Tensor<Dim2>,
        result: &mut Self::Tensor<Dim2>,
    ) {
        self.general_program.sigmoid_error(&self.queue, output, out_error, result)
    }

    fn relu(&self, leak: Self::Float, activation: &Self::Tensor<Dim2>, output: &mut Self::Tensor<Dim2>) {
        todo!()
    }

    fn relu_error(
        &self,
        leak: Self::Float,
        activation: &Self::Tensor<Dim2>,
        out_error: &Self::Tensor<Dim2>,
        result: &mut Self::Tensor<Dim2>,
    ) {
        todo!()
    }

    fn softmax(&self, activation: &Self::Tensor<Dim2>, output: &mut Self::Tensor<Dim2>) {
        Softmax::get_or_create(
            &self.context,
            &self.cache,
            self.vec_width,
            output.dims().cols(),
            output.buffer_dims().cols(),
        )
        .unwrap()
        .softmax(&self.queue, activation, output)
    }

    fn softmax_error(
        &self,
        output: &Self::Tensor<Dim2>,
        out_error: &Self::Tensor<Dim2>,
        result: &mut Self::Tensor<Dim2>,
    ) {
        todo!()
    }

    fn mean_squared_error(
        &self,
        output: &Self::Tensor<Dim2>,
        expected: &Self::Tensor<Dim2>,
        result: &mut Self::Tensor<Dim1>,
        result_deriv: &mut Self::Tensor<Dim2>,
    ) {
        MSEProgram::get_or_create(
            &self.context,
            &self.cache,
            self.vec_width,
            output.dims().cols(),
            output.buffer_dims().cols(),
        )
        .unwrap()
        .mean_squared_error(&self.queue, output, expected, result, result_deriv);
    }

    fn flush(&self) {
        // enqueue a barrier that requires all previous commands finish before the next call
        wrap_cl_error!(
            unsafe { self.queue.enqueue_barrier_with_wait_list(&[]) },
            "Error calling enqueue_barrier_with_wait_list"
        )
        .unwrap();
        wrap_cl_error!(self.queue.flush(), "Error flushing queue").unwrap();
    }

    fn sync(&self) {
        wrap_cl_error!(self.queue.finish(), "Error finishing queue").unwrap()
    }

    fn accum_confusion_matrix_multiclass(
        &self,
        matrix: &mut Self::Tensor<Dim2>,
        output: &Self::Tensor<Dim2>,
        expected: &Self::Tensor<Dim2>,
    ) {
        self.scoring_program
            .accum_multiclass_confusion_matrix(&self.context, &self.queue, matrix, output, expected);
    }
}
