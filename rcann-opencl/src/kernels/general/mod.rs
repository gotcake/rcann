#[cfg(test)]
mod test;

use crate::kernels::BUFFER_BLOCK_SIZE;
use crate::tensor::event_list::EventList;
use crate::tensor::{OclFloat, OclTensor, OclTensor1, OclTensor2};
use crate::util::{ocl_program, VecWidth};
use opencl3::kernel::ExecuteKernel;
use rcann::tensor::{Dims, ITensor};

ocl_program! {
    name = GeneralProgram,
    source = "general.cl",
    generic_args = <T: OclFloat>,
    compile_params = (
        vec_width: VecWidth,
        vec_per_thread: usize,
    ),
    validation = {
        validate!(BUFFER_BLOCK_SIZE % (*vec_width as usize * *vec_per_thread) == 0, "BUFFER_BLOCK_SIZE must be a multiple of vec_width * vec_per_thread");
    },
    defines = {
        FLOAT_BITS = T::BITS,
        VECTOR_WIDTH = *vec_width,
        VECTOR_PER_THREAD = *vec_per_thread,
    },
    kernels = {
        sigmoid {
            call_params = (
                activation: &OclTensor2<T>,
                output: &mut OclTensor2<T>,
            ),
            pre = {
                let unit_width = *vec_width as usize * *vec_per_thread;
                let n = activation.buffer_len();
            },
            validation = {
                assert_eq!(activation.buffer_dims(), output.buffer_dims());
                assert_eq!(n % unit_width, 0);
            },
            inputs = [activation],
            outputs = [output],
            kernel_args = [
                activation.buffer(),
                output.buffer(),
            ],
            global_dims = [n / unit_width],
        },
        sigmoid_error {
            call_params = (
                output: &OclTensor2<T>,
                error: &OclTensor2<T>,
                result: &mut OclTensor2<T>,
            ),
            pre = {
                let unit_width = *vec_width as usize * *vec_per_thread;
                let n = output.buffer_len();
            },
            validation = {
                assert_eq!(output.buffer_dims(), error.buffer_dims());
                assert_eq!(output.buffer_dims(), result.buffer_dims());
                assert_eq!(n % unit_width, 0);
            },
            inputs = [output, error],
            outputs = [result],
            kernel_args = [
                output.buffer(),
                error.buffer(),
                result.buffer(),
            ],
            global_dims = [n / unit_width],
        },
        add_assign {
            generic_args = <D: Dims>,
            call_params = (
                alpha: T,
                input: &OclTensor<T, D>,
                beta: T,
                output: &mut OclTensor<T, D>,
            ),
            pre = {
                let unit_width = *vec_width as usize * *vec_per_thread;
                let n = input.buffer_len();
            },
            validation = {
                assert_eq!(input.buffer_dims(), output.buffer_dims());
                assert_eq!(n % unit_width, 0);
            },
            inputs = [input, output],
            outputs = [output],
            kernel_args = [
                &alpha,
                &beta,
                input.buffer(),
                output.buffer(),
            ],
            global_dims = [n / unit_width],
        },
        column_sum {
            call_params = (
                alpha: T,
                input: &OclTensor2<T>,
                beta: T,
                output: &mut OclTensor1<T>,
            ),
            pre = {
                let n = output.buffer_len();
                let rows = input.dims().rows();
                let cols = input.dims().cols();
            },
            validation = {
                assert_eq!(input.buffer_dims().cols(), n);
                assert_eq!(n % *vec_width as usize, 0);
            },
            inputs = [input, output],
            outputs = [output],
            kernel_args = [
                &(rows as u32),
                &(cols as u32),
                &((n / *vec_width as usize) as u32),
                &alpha,
                &beta,
                input.buffer(),
                output.buffer(),
            ],
            global_dims = [n / *vec_width as usize],
        },
    },
}
