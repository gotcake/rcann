#[cfg(test)]
mod test;

use crate::tensor::{OclTensor, OclTensor1, OclTensor2};
use crate::{
    format_c_defines,
    util::{self, Result},
    wrap_cl_error,
};
use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::program::Program;
use opencl3::types::cl_uint;
use rcann::tensor::{Dim2, Dims, ITensor};
use crate::tensor::event_list::EventList;

#[derive(Debug)]
pub struct GeneralKernels {
    #[allow(unused)]
    program: Program,
    sigmoid: Kernel,
    sigmoid_error: Kernel,
    add_assign: Kernel,
    column_sum: Kernel,
}

pub mod constants {
    pub const BLOCK_SIZE: usize = 16;
    pub const PER_THREAD: usize = 4;
}

impl GeneralKernels {
    pub fn new(context: &Context) -> Result<Self> {
        let mut code = format_c_defines!(
            "PER_THREAD" => constants::PER_THREAD,
        );
        code.push_str(include_str!("general.cl"));
        let program = util::create_program(context, code.as_ref(), "")?;
        let sigmoid = util::create_kernel(&program, "sigmoid")?;
        let sigmoid_error = util::create_kernel(&program, "sigmoid_error")?;
        let add_assign = util::create_kernel(&program, "add_assign")?;
        let column_sum = util::create_kernel(&program, "column_sum")?;
        Ok(Self {
            program,
            sigmoid,
            sigmoid_error,
            add_assign,
            column_sum,
        })
    }

    pub fn sigmoid(
        &self,
        queue: &CommandQueue,
        activation: &OclTensor2<f32>,
        output: &mut OclTensor2<f32>,
    ) -> Result<()> {
        assert_eq!(activation.buffer_dims(), output.buffer_dims());
        let n = activation.buffer_len();
        assert_eq!(n % constants::BLOCK_SIZE, 0);
        let mut exec = ExecuteKernel::new(&self.sigmoid);
        unsafe {
            exec.set_arg(activation.buffer()).set_arg(output.buffer());
        }
        exec.set_local_work_size(constants::BLOCK_SIZE / constants::PER_THREAD)
            .set_global_work_size(n / constants::PER_THREAD);
        exec.set_event_wait_list(activation.deps().as_slice());
        let kernel_evt = wrap_cl_error!(
            unsafe { exec.enqueue_nd_range(queue) },
            "Failed to enqueue sigmoid kernel"
        )?;
        output.set_deps(EventList::from_event(kernel_evt));
        Ok(())
    }

    pub fn sigmoid_error(
        &self,
        queue: &CommandQueue,
        output: &OclTensor2<f32>,
        error: &OclTensor2<f32>,
        result: &mut OclTensor2<f32>,
    ) -> Result<()> {
        assert_eq!(result.buffer_dims(), output.buffer_dims());
        assert_eq!(result.buffer_dims(), error.buffer_dims());
        let n = output.buffer_len();
        assert_eq!(n % constants::BLOCK_SIZE, 0);
        let mut exec = ExecuteKernel::new(&self.sigmoid_error);
        unsafe {
            exec.set_arg(output.buffer())
                .set_arg(error.buffer())
                .set_arg(result.buffer());
        }
        exec.set_local_work_size(constants::BLOCK_SIZE / constants::PER_THREAD)
            .set_global_work_size(n / constants::PER_THREAD);
        let deps = EventList::concat([output.deps(), error.deps()]);
        exec.set_event_wait_list(deps.as_slice());
        let kernel_evt = wrap_cl_error!(
            unsafe { exec.enqueue_nd_range(queue) },
            "Failed to enqueue sigmoid_error kernel"
        )?;
        result.set_deps(EventList::from_event(kernel_evt));
        Ok(())
    }

    pub fn add_assign<D: Dims>(
        &self,
        queue: &CommandQueue,
        alpha: f32,
        input: &OclTensor<f32, D>,
        beta: f32,
        output: &mut OclTensor<f32, D>,
    ) -> Result<()> {
        assert_eq!(input.buffer_dims(), output.buffer_dims());
        let n = input.buffer_len();
        assert_eq!(n % constants::BLOCK_SIZE, 0);
        let mut exec = ExecuteKernel::new(&self.add_assign);
        unsafe {
            exec.set_arg(&alpha)
                .set_arg(&beta)
                .set_arg(input.buffer())
                .set_arg(output.buffer());
        }
        exec.set_local_work_size(constants::BLOCK_SIZE / constants::PER_THREAD)
            .set_global_work_size(n / constants::PER_THREAD);
        let deps = EventList::concat([output.deps(), input.deps()]);
        exec.set_event_wait_list(deps.as_slice());
        let kernel_evt = wrap_cl_error!(
            unsafe { exec.enqueue_nd_range(queue) },
            "Failed to enqueue add_assign kernel"
        )?;
        output.set_deps(EventList::from_event(kernel_evt));
        Ok(())
    }

    pub fn column_sum(
        &self,
        queue: &CommandQueue,
        alpha: f32,
        input: &OclTensor2<f32>,
        beta: f32,
        output: &mut OclTensor1<f32>,
    ) -> Result<()> {
        assert_eq!(input.buffer_dims().cols(), output.buffer_len());
        let n = output.buffer_len();
        assert_eq!(n % constants::BLOCK_SIZE, 0);
        let mut exec = ExecuteKernel::new(&self.column_sum);
        let &Dim2(rows, cols) = input.dims();
        unsafe {
            exec.set_arg(&(rows as cl_uint))
                .set_arg(&(cols as cl_uint))
                .set_arg(&(input.buffer_dims().cols() as cl_uint))
                .set_arg(&alpha)
                .set_arg(&beta)
                .set_arg(input.buffer())
                .set_arg(output.buffer());
        }
        exec.set_local_work_size(constants::BLOCK_SIZE / constants::PER_THREAD)
            .set_global_work_size(n / constants::PER_THREAD);
        let deps = EventList::concat([output.deps(), input.deps()]);
        exec.set_event_wait_list(deps.as_slice());
        let kernel_evt = wrap_cl_error!(
            unsafe { exec.enqueue_nd_range(queue) },
            "Failed to enqueue column_sum kernel"
        )?;
        output.set_deps(EventList::from_event(kernel_evt));
        Ok(())
    }
}
