use crate::error::Error;
use crate::tensor::OclTensor;
use crate::util::{self, next_multiple, Result};
use crate::wrap_cl_error;
use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::program::Program;
use opencl3::types::cl_uint;
use rcann::tensor::{Dim2, ITensor};

#[derive(Debug)]
pub struct ZeroPaddingKernel {
    #[allow(unused)]
    program: Program,
    kernel: Kernel,
}

mod kernel_const {
    pub const BLOCK_SIZE: usize = 16;
}

impl ZeroPaddingKernel {
    pub fn create(context: &Context) -> Result<Self> {
        let program = util::create_program(context, include_str!("zero_padding.cl"), "")?;
        let kernel = util::create_kernel(&program, "zero_padding")?;
        Ok(Self { program, kernel })
    }

    fn zero_padding(&self, queue: &CommandQueue, tensor: &mut OclTensor<f32, Dim2>) -> Result<()> {
        if tensor.dims() == tensor.buffer_dims() {
            return Ok(());
        }
        let &Dim2(rows, _cols) = tensor.dims();
        let &Dim2(buff_rows, buff_cols) = tensor.buffer_dims();

        let mut exec = ExecuteKernel::new(&self.kernel);
        unsafe {
            exec.set_arg(&(rows as cl_uint))
                .set_arg(&(buff_rows as cl_uint))
                .set_arg(&(buff_cols as cl_uint))
                .set_arg(tensor.buffer())
        };
        let n = next_multiple(buff_cols, kernel_const::BLOCK_SIZE);
        exec.set_local_work_size(kernel_const::BLOCK_SIZE)
            .set_global_work_size(n);
        exec.set_event_wait_list(tensor.get_deps().as_slice());

        let kernel_evt = wrap_cl_error!(
            unsafe { exec.enqueue_nd_range(queue) },
            "Failed to enqueue zero_padding kernel"
        )?;
        tensor.set_dep(kernel_evt);
        Ok(())
    }
}
