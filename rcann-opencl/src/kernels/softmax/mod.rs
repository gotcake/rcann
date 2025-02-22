use crate::tensor::{OclFloat, OclTensor2};
use crate::util::*;
#[cfg(test)]
mod test;

ocl_program! {
    name = Softmax,
    source = "softmax2.cl",
    generic_args = <T: OclFloat>,
    compile_params = (
        vec_width: u8,
        cols: usize,
        row_stride: usize,
    ),
    validation = {
        validate!(is_valid_vec_width(*vec_width), "Invalid vector width");
        validate!(*row_stride % *vec_width as usize == 0, "Misaligned row stride");
    },
    defines = {
        FLOAT_BITS = T::BITS,
        VEC_WIDTH = vec_width,
        COLS = cols,
        ROW_STRIDE = row_stride,
        VEC_COLS = *cols / *vec_width as usize,
        VEC_COLS_REM = *cols % *vec_width as usize,
    },
    kernels = {
        softmax {
            call_params = (
                activation: &OclTensor2<T>,
                output: &mut OclTensor2<T>
            ),
            validation = {
                assert_eq!(activation.dims(), output.dims());
                assert_eq!(activation.buffer_dims().cols(), *row_stride);
                assert_eq!(output.buffer_dims().cols(), *row_stride);
            },
            inputs = [activation],
            outputs = [output],
            kernel_args = [
                &(activation.dims().rows() as u32),
                activation.buffer(),
                output.buffer(),
            ],
            global_dims = [next_multiple(activation.dims().rows(), 16)],

        },
    },
}
