#[cfg(test)]
mod test;

use crate::error::Error;
use crate::tensor::{OclTensor, OclTensor2};
use crate::util::{next_multiple, Result};
use crate::{format_c_defines, util, wrap_cl_error};
use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::program::Program;
use opencl3::types::cl_uint;
use rcann::tensor::{Dim2, ITensor};

#[derive(Debug)]
pub struct TransposeKernel {
    #[allow(unused)]
    program: Program,
    kernel: Kernel,
}

pub mod constants {
    pub const BLOCK_SIZE: usize = 16;
}

impl TransposeKernel {
    pub fn create(context: &Context) -> Result<Self> {
        let mut code = format_c_defines!(
            "BLOCK_SIZE" => constants::BLOCK_SIZE,
        );
        code.push_str(include_str!("transpose.cl"));
        let program = util::create_program(context, code.as_ref(), "")?;
        let kernel = util::create_kernel(&program, "transpose")?;
        Ok(TransposeKernel { program, kernel })
    }
    pub fn transpose(&self, queue: &CommandQueue, input: &OclTensor2<f32>, output: &mut OclTensor2<f32>) -> Result<()> {
        assert_eq!(input.dims(), &output.dims().transposed());
        let &Dim2(rows, cols) = input.dims();
        let &Dim2(out_buff_rows, out_buff_cols) = output.buffer_dims();
        let in_row_stride = input.buffer_dims().cols();
        //let m = next_multiple(in_rows.max(out_rows), constants::BLOCK_SIZE);
        //let n = next_multiple(in_cols.max(out_cols), constants::BLOCK_SIZE);
        let m = out_buff_cols;
        let n = out_buff_rows;
        assert_eq!(m % constants::BLOCK_SIZE, 0);
        assert_eq!(n % constants::BLOCK_SIZE, 0);
        let mut exec = ExecuteKernel::new(&self.kernel);
        unsafe {
            exec.set_arg(&(rows as cl_uint))
                .set_arg(&(cols as cl_uint))
                .set_arg(&(in_row_stride as cl_uint))
                .set_arg(&(out_buff_rows as cl_uint))
                .set_arg(&(out_buff_cols as cl_uint))
                .set_arg(input.buffer())
                .set_arg(output.buffer());
        }
        exec.set_event_wait_list(input.get_deps().as_slice());
        exec.set_local_work_sizes(&[constants::BLOCK_SIZE, constants::BLOCK_SIZE])
            .set_global_work_sizes(&[m, n]);
        let kernel_evt = wrap_cl_error!(
            unsafe { exec.enqueue_nd_range(queue) },
            "Failed to enqueue transpose kernel"
        )?;
        output.set_dep(kernel_evt);
        Ok(())
    }
}
