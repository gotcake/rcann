#[cfg(test)]
mod test;

use crate::tensor::event_list::EventList;
use crate::tensor::{OclFloat, OclTensor1, OclTensor2};
use crate::util::{next_multiple, ocl_program};
use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::kernel::ExecuteKernel;
use rcann::tensor::{Dim1, Dim2, ITensor};

ocl_program! {
    name = ScoringProgram,
    source = "scoring.cl",
    generic_args = <T: OclFloat>,
    defines = {
        FLOAT_BITS = T::BITS,
    },
    kernels = {
        compute_confusion_matrix_indices {
            call_params = (
                output: &OclTensor2<T>,
                expected: &OclTensor2<T>,
                index_buffer: &mut OclTensor1<u32>,
            ),
            pre = {
                let rows = output.dims().rows();
                let classes = output.dims().cols();
            },
            validation = {
                assert_eq!(output.dims(), expected.dims());
                assert_eq!(rows * 2, index_buffer.len());
            },
            inputs = [output, expected],
            outputs = [index_buffer],
            kernel_args = [
                &(rows as u32),
                &(classes as u32),
                &(output.buffer_dims().cols() as u32),
                &(expected.buffer_dims().cols() as u32),
                output.buffer(),
                expected.buffer(),
                index_buffer.buffer(),
            ],
            global_dims = [next_multiple(rows, 16)],
        },
        inc_by_indices {
            call_params = (
                matrix: &mut OclTensor2<T>,
                index_buffer: &OclTensor1<u32>,
            ),
            pre = {
                let rows = index_buffer.len() / 2;
                let classes = matrix.dims().rows();
            },
            validation = {
                assert_eq!(index_buffer.len() % 2, 0);
                assert_eq!(matrix.dims().cols(), classes);
            },
            inputs = [index_buffer, matrix],
            outputs = [matrix],
            kernel_args = [
                &(rows as u32),
                &(classes as u32),
                &(matrix.buffer_dims().cols() as u32),
                index_buffer.buffer(),
                matrix.buffer(),
            ],
            global_dims = [next_multiple(classes, 16)],
        },
    },
}

impl<T: OclFloat> ScoringProgram<T> {
    pub(crate) fn accum_multiclass_confusion_matrix(
        &self,
        context: &Context,
        queue: &CommandQueue,
        matrix: &mut OclTensor2<T>,
        output: &OclTensor2<T>,
        expected: &OclTensor2<T>,
    ) {
        let &Dim2(rows, n) = output.dims();
        assert_eq!(expected.dims(), output.dims());
        assert_eq!(matrix.dims(), &Dim2(n, n));

        let mut index_buffer = unsafe { OclTensor1::uninit(context, Dim1(rows * 2)).unwrap() };
        self.compute_confusion_matrix_indices(queue, output, expected, &mut index_buffer);
        self.inc_by_indices(queue, matrix, &index_buffer);
    }
}