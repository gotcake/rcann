#[cfg(test)]
mod test;

use crate::tensor::event_list::EventList;
use crate::tensor::{OclFloat, OclTensor1, OclTensor2};
use crate::util::*;
use opencl3::kernel::ExecuteKernel;
use rcann::tensor::ITensor;


ocl_program!(
    name = MSEProgram,
    source = "mean_squared_error2.cl",
    generic_args = <T: OclFloat>,
    compile_params = (
        vec_width: VecWidth,
        cols: usize,
        row_stride: usize,
    ),
    validation = {
        validate!(*cols > 0, "cols must be positive");
        validate!(*row_stride > 0, "row_stride must be positive");
        validate!(*row_stride % *vec_width as usize == 0, "Misaligned row stride");
    },
    defines = {
        FLOAT_BITS = T::BITS,
        VEC_WIDTH = *vec_width as usize,
        COLS = *cols,
        ROW_STRIDE = *row_stride,
        VEC_COLS = *cols / *vec_width as usize,
        VEC_COLS_REM = *cols % *vec_width as usize,
    },
    kernels = {
        mean_squared_error {
            call_params = (
                output: &OclTensor2<T>,
                expected: &OclTensor2<T>,
                result: &mut OclTensor1<T>,
                result_deriv: &mut OclTensor2<T>,
            ),
            validation = {
                assert_eq!(output.dims(), expected.dims());
                assert_eq!(output.dims(), result_deriv.dims());
                assert_eq!(result.dims().major(), output.dims().rows());
                assert_eq!(output.buffer_dims().cols(), *row_stride);
                assert_eq!(expected.buffer_dims().cols(), *row_stride);
                assert_eq!(result_deriv.buffer_dims().cols(), *row_stride);
            },
            inputs = [output, expected],
            outputs = [result, result_deriv],
            kernel_args = [
                &(output.dims().rows() as u32),
                output.buffer(),
                expected.buffer(),
                result.buffer(),
                result_deriv.buffer(),
            ],
            global_dims = [next_multiple(output.dims().rows(), 16)],
        },
    },
);
