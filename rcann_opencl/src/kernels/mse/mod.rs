#[cfg(test)]
mod test;

use crate::tensor::event_list::EventList;
use crate::tensor::{OclTensor1, OclTensor2};
use crate::util::{next_multiple, Result};
use crate::{format_c_defines, util, wrap_cl_error};
use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::program::Program;
use opencl3::types::cl_uint;
use rcann::tensor::{Dim2, ITensor};

#[derive(Debug)]
pub struct MSEKernel {
    #[allow(unused)]
    program: Program,
    kernel: Kernel,
}

pub mod constants {
    pub const VECTOR_WIDTH: usize = 16;
    pub const GROUP_SIZE: usize = 16;
}

impl MSEKernel {
    pub fn new(context: &Context) -> Result<Self> {
        let mut code = format_c_defines!(
            "FLOAT_BITS" => 32,
            "VECTOR_WIDTH" => constants::VECTOR_WIDTH,
        );
        code.push_str(include_str!("../types.cl"));
        code.push_str("\n");
        code.push_str(include_str!("mean_squared_error.cl"));
        let program = util::create_program(context, code.as_ref(), "")?;
        let kernel = util::create_kernel(&program, "mean_squared_error")?;
        Ok(Self { program, kernel })
    }

    pub fn mean_squared_error(
        &self,
        queue: &CommandQueue,
        output: &OclTensor2<f32>,
        expected: &OclTensor2<f32>,
        result: &mut OclTensor1<f32>,
        result_deriv: &mut OclTensor2<f32>,
    ) -> Result<()> {
        assert_eq!(output.buffer_dims(), expected.buffer_dims());
        assert_eq!(result_deriv.buffer_dims(), output.buffer_dims());
        assert_eq!(result.buffer_len(), output.buffer_dims().rows());
        let &Dim2(rows, cols) = output.dims();
        let buff_cols = output.buffer_dims().cols();
        assert_eq!(buff_cols % constants::VECTOR_WIDTH, 0);

        let mut exec = ExecuteKernel::new(&self.kernel);
        unsafe {
            exec.set_arg(&(rows as cl_uint))
                .set_arg(&((cols) as cl_uint))
                .set_arg(&((buff_cols) as cl_uint))
                .set_arg(output.buffer())
                .set_arg(expected.buffer())
                .set_arg(result.buffer())
                .set_arg(result_deriv.buffer());
        }
        let n = next_multiple(rows, constants::GROUP_SIZE);
        exec.set_global_work_size(n);
        let deps = EventList::concat([output.deps(), expected.deps()]);
        exec.set_event_wait_list(deps.as_slice());
        let kernel_evt = wrap_cl_error!(
            unsafe { exec.enqueue_nd_range(queue) },
            "Failed to enqueue mean_squared_error kernel"
        )?;
        let events = EventList::from_event(kernel_evt);
        result.set_deps(events.clone());
        result_deriv.set_deps(events);
        Ok(())
    }
}
