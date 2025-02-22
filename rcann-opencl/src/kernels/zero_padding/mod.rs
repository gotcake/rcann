use crate::tensor::event_list::EventList;
use crate::tensor::{OclFloat, OclTensor};
use crate::util::{next_multiple, ocl_program};
use opencl3::kernel::ExecuteKernel;
use rcann::tensor::{Dim2, ITensor};

ocl_program! {
    name = ZeroPadProgram,
    source = "zero_padding.cl",
    generic_args = <T: OclFloat>,
    compile_params = (
        block_size: usize,
    ),
    validation = {
        validate!(is_power_of_two(*block_size), "block_size must be a power of 2");
        //validate!(BUFFER_BLOCK_SIZE % *block_size == 0, "BUFFER_BLOCK_SIZE must be a multiple of block_size");
    },
    defines = {
        FLOAT_BITS = T::BITS,
        BLOCK_SIZE = block_size,
    },
    kernels = {
        zero_padding {
            call_params = (
                tensor: &OclTensor<T, Dim2>
            ),
            pre = {
                let rows = tensor.dims().rows();
                let cols = tensor.dims().cols();
                let buff_rows = tensor.buffer_dims().rows();
                let buff_cols = tensor.buffer_dims().cols();
            },
            inputs = [tensor],
            outputs = [tensor],
            kernel_args = [
                &(rows as u32),
                &(cols as u32),
                &(buff_rows as u32),
                &(buff_cols as u32),
                tensor.buffer(),
            ],
            global_dims = [next_multiple(usize::max(buff_rows, buff_cols), *block_size)],
            local_dims = [*block_size],
        },
    },
}

// TODO: tests