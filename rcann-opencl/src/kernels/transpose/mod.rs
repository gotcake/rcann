#[cfg(test)]
mod test;

use crate::kernels::BUFFER_BLOCK_SIZE;
use crate::tensor::event_list::EventList;
use crate::tensor::{OclFloat, OclTensor2};
use crate::util::ocl_program;
use opencl3::kernel::ExecuteKernel;
use rcann::tensor::ITensor;

ocl_program! {
    name = TransposeProgram,
    source = "transpose.cl",
    generic_args = <T: OclFloat>,
    compile_params = (
        block_size: usize,
    ),
    validation = {
        validate!(is_power_of_two(*block_size), "block_size must be a power of 2");
        validate!(BUFFER_BLOCK_SIZE % *block_size == 0, "BUFFER_BLOCK_SIZE must be a multiple of block_size");
    },
    defines = {
        FLOAT_BITS = T::BITS,
        BLOCK_SIZE = block_size,
    },
    kernels = {
        transpose {
            call_params = (
                input: &OclTensor2<T>,
                output: &mut OclTensor2<T>,
            ),
            pre = {
                let rows = input.dims().rows();
                let cols = input.dims().cols();
                let m = output.buffer_dims().cols();
                let n = output.buffer_dims().rows();
                let in_row_stride = input.buffer_dims().cols();
            },
            validation = {
                assert_eq!(m % *block_size, 0);
                assert_eq!(n % *block_size, 0);
            },
            inputs = [input],
            outputs = [output],
            kernel_args = [
                &(rows as u32),
                &(cols as u32),
                &(in_row_stride as u32),
                &(n as u32),
                &(m as u32),
                input.buffer(),
                output.buffer(),
            ],
            global_dims = [m, n],
            local_dims = [*block_size, *block_size],
        },
    },
}
