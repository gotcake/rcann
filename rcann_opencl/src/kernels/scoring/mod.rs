#[cfg(test)]
mod test;

use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::program::Program;
use opencl3::types::cl_uint;
use rcann::tensor::{Dim2, ITensor};
use crate::{format_c_defines, util, wrap_cl_error};
use crate::tensor::event_list::EventList;
use crate::tensor::OclTensor2;
use crate::util::{next_multiple, Result};

pub struct ScoringKernels {
    #[allow(unused)]
    program: Program,
    accum_multiclass_confusion_matrix: Kernel,
}

pub mod constants {
    pub const GLOBAL_WORK_SIZE_MULTIPLE: usize = 16;
}

impl ScoringKernels {
    pub fn create(context: &Context) -> Result<Self> {
        let mut code = format_c_defines!();
        code.push_str(include_str!("scoring.cl"));
        let program = util::create_program(context, code.as_ref(), "")?;
        let accum_multiclass_confusion_matrix = util::create_kernel(&program, "accum_multiclass_confusion_matrix")?;
        Ok(Self {
            program,
            accum_multiclass_confusion_matrix,
        })
    }

    pub fn accum_multiclass_confusion_matrix(
        &self,
        queue: &CommandQueue,
        matrix: &mut OclTensor2<f32>,
        output: &OclTensor2<f32>,
        expected: &OclTensor2<f32>
    ) -> Result<()> {
        let rows = output.dims().rows();
        let n = matrix.dims().rows();
        assert_eq!(output.dims(), &Dim2(rows, n));
        assert_eq!(expected.dims(), &Dim2(rows, n));
        assert_eq!(matrix.dims(), &Dim2(n, n));

        let mut exec = ExecuteKernel::new(&self.accum_multiclass_confusion_matrix);
        unsafe {
            exec.set_arg(&(rows as cl_uint))
                .set_arg(&(n as cl_uint))
                .set_arg(&(output.buffer_dims().cols() as cl_uint))
                .set_arg(&(expected.buffer_dims().cols() as cl_uint))
                .set_arg(&(matrix.buffer_dims().cols() as cl_uint))
                .set_arg(output.buffer())
                .set_arg(expected.buffer())
                .set_arg(matrix.buffer());
        }
        
        let deps = EventList::concat([output.deps(), expected.deps(), matrix.deps()]);
        exec.set_event_wait_list(deps.as_slice());
        
        let global_work_size = next_multiple(rows, constants::GLOBAL_WORK_SIZE_MULTIPLE);
        exec.set_local_work_size(constants::GLOBAL_WORK_SIZE_MULTIPLE).set_global_work_size(global_work_size);

        let kernel_evt = wrap_cl_error!(
            unsafe { exec.enqueue_nd_range(queue) },
            "Failed to enqueue accum_multiclass_confusion_matrix kernel"
        )?;
        matrix.set_deps(EventList::from_event(kernel_evt));
        Ok(())

    }
}