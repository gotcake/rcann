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
use rcann::tensor::{Dim1, Dim2, ITensor};

#[derive(Debug)]
pub struct ScoringKernels {
    #[allow(unused)]
    program: Program,
    compute_confusion_matrix_indices: Kernel,
    inc_by_indices: Kernel,
}

pub mod constants {
    pub const GLOBAL_WORK_SIZE_MULTIPLE: usize = 16;
}

impl ScoringKernels {
    pub fn create(context: &Context) -> Result<Self> {
        let mut code = format_c_defines!();
        code.push_str(include_str!("scoring.cl"));
        let program = util::create_program(context, code.as_ref(), "")?;
        let compute_confusion_matrix_indices = util::create_kernel(&program, "compute_confusion_matrix_indices")?;
        let inc_by_indices = util::create_kernel(&program, "inc_by_indices")?;
        Ok(Self {
            program,
            compute_confusion_matrix_indices,
            inc_by_indices,
        })
    }

    pub fn accum_multiclass_confusion_matrix(
        &self,
        context: &Context,
        queue: &CommandQueue,
        matrix: &mut OclTensor2<f32>,
        output: &OclTensor2<f32>,
        expected: &OclTensor2<f32>,
    ) -> Result<()> {
        let &Dim2(rows, n) = output.dims();
        assert_eq!(expected.dims(), output.dims());
        assert_eq!(matrix.dims(), &Dim2(n, n));

        let mut index_buffer = unsafe { OclTensor1::uninit(context, Dim1(rows * 2))? };
        self.compute_confusion_matrix_indices(queue, output, expected, &mut index_buffer)?;

        self.inc_by_indices(queue, rows, n, matrix, &index_buffer)?;

        Ok(())
    }

    fn compute_confusion_matrix_indices(
        &self,
        queue: &CommandQueue,
        output: &OclTensor2<f32>,
        expected: &OclTensor2<f32>,
        index_buffer: &mut OclTensor1<u32>,
    ) -> Result<()> {
        let &Dim2(rows, n) = output.dims();

        let mut exec = ExecuteKernel::new(&self.compute_confusion_matrix_indices);
        unsafe {
            exec.set_arg(&(rows as cl_uint))
                .set_arg(&(n as cl_uint))
                .set_arg(&(output.buffer_dims().cols() as cl_uint))
                .set_arg(&(expected.buffer_dims().cols() as cl_uint))
                .set_arg(output.buffer())
                .set_arg(expected.buffer())
                .set_arg(index_buffer.buffer());
        }

        let deps = EventList::concat([output.deps(), expected.deps()]);
        exec.set_event_wait_list(deps.as_slice());

        let global_work_size = next_multiple(rows, constants::GLOBAL_WORK_SIZE_MULTIPLE);
        exec.set_global_work_size(global_work_size);

        let kernel_evt = wrap_cl_error!(
            unsafe { exec.enqueue_nd_range(queue) },
            "Failed to enqueue compute_confusion_matrix_indices kernel"
        )?;

        index_buffer.set_deps(EventList::from_event(kernel_evt));
        Ok(())
    }

    fn inc_by_indices(
        &self,
        queue: &CommandQueue,
        rows: usize,
        n: usize,
        matrix: &mut OclTensor2<f32>,
        index_buffer: &OclTensor1<u32>,
    ) -> Result<()> {
        let mut exec = ExecuteKernel::new(&self.inc_by_indices);
        unsafe {
            exec.set_arg(&(rows as cl_uint))
                .set_arg(&(n as cl_uint))
                .set_arg(&(matrix.buffer_dims().cols() as cl_uint))
                .set_arg(index_buffer.buffer())
                .set_arg(matrix.buffer());
        }

        let deps = EventList::concat([index_buffer.deps(), matrix.deps()]);
        exec.set_event_wait_list(deps.as_slice());

        let global_work_size = next_multiple(n, constants::GLOBAL_WORK_SIZE_MULTIPLE);
        exec.set_global_work_size(global_work_size);

        let kernel_evt = wrap_cl_error!(
            unsafe { exec.enqueue_nd_range(queue) },
            "Failed to enqueue inc_by_indices kernel"
        )?;

        matrix.set_deps(EventList::from_event(kernel_evt));
        Ok(())
    }
}
