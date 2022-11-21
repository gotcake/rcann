#[cfg(test)]
mod test;

use crate::error::Error;
use crate::tensor::OclTensor;
use crate::util::Result;
use crate::{format_c_defines, util, wrap_cl_error};
use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::program::Program;
use opencl3::types::cl_uint;
use rcann::tensor::Dim2;

#[derive(Debug)]
pub struct GemmKernel {
    #[allow(unused)]
    program: Program,
    kernel: Kernel,
}

#[allow(unused)]
pub mod constants {
    pub const TILE_SIZE: usize = 16;
    pub const WORK_PER_THREAD: usize = 4;
    pub const REDUCED_TILE_SIZE: usize = TILE_SIZE / WORK_PER_THREAD;

    pub const WIDTH: usize = 4;
    pub const TSM: usize = 16;
    pub const TSN: usize = 16;
    pub const TSK: usize = 16;
    pub const WPTM: usize = 8;
    pub const WPTN: usize = 8;
    pub const RTSM: usize = TSM / WPTM;
    pub const RTSN: usize = TSN / WPTN;
    pub const LPTA: usize = (TSK * WPTM * WPTN) / TSN;
    pub const LPTB: usize = (TSK * WPTM * WPTN) / TSM;
}

impl GemmKernel {
    pub fn new(context: &Context) -> Result<GemmKernel> {
        let mut code = format_c_defines!(
            "TILE_SIZE" => constants::TILE_SIZE,
            "WORK_PER_THREAD" => constants::WORK_PER_THREAD,
            "REDUCED_TILE_SIZE" => constants::REDUCED_TILE_SIZE,
        );
        /*let mut code = format_c_defines!(
            "WIDTH" => kernel_const::WIDTH,
            "TSM" => kernel_const::TSM,
            "TSN" => kernel_const::TSN,
            "TSK" => kernel_const::TSK,
            "WPTM" => kernel_const::WPTM,
            "WPTN" => kernel_const::WPTN,
            "RTSM" => kernel_const::RTSM,
            "RTSN" => kernel_const::RTSN,
            "LPTA" => kernel_const::LPTA,
            "LPTB" => kernel_const::LPTB,
        );*/
        code.push_str(include_str!("gemm.cl"));
        let program = util::create_program(context, code.as_ref(), "")?;
        let kernel = util::create_kernel(&program, "sgemm3")?;
        Ok(GemmKernel { program, kernel })
    }

    pub fn gemm(
        &self,
        queue: &CommandQueue,
        alpha: f32,
        a: &OclTensor<f32, Dim2>,
        b: &OclTensor<f32, Dim2>,
        beta: f32,
        c: &mut OclTensor<f32, Dim2>,
    ) -> Result<()> {
        let &Dim2(m, k) = a.buffer_dims();
        let &Dim2(_, n) = b.buffer_dims();
        assert_eq!(b.buffer_dims(), &Dim2(k, n));
        assert_eq!(c.buffer_dims(), &Dim2(m, n));
        assert_eq!(m % constants::TILE_SIZE, 0);
        assert_eq!(n % constants::TILE_SIZE, 0);
        assert_eq!(k % constants::TILE_SIZE, 0);
        let mut exec = ExecuteKernel::new(&self.kernel);
        unsafe {
            exec.set_arg(&(m as cl_uint)) // M
                .set_arg(&(k as cl_uint)) // K
                .set_arg(&(n as cl_uint)) // N
                .set_arg(&alpha) // ALPHA
                .set_arg(a.buffer()) // A
                .set_arg(b.buffer()) // B
                .set_arg(&beta) // BETA
                .set_arg(c.buffer_mut()) // C
        };
        exec.set_local_work_sizes(&[constants::TILE_SIZE, constants::REDUCED_TILE_SIZE]) // TODO?
            .set_global_work_sizes(&[m, n / constants::WORK_PER_THREAD]);
        let deps = [a.get_deps(), b.get_deps(), c.get_deps()].concat();
        exec.set_event_wait_list(deps.as_slice());
        let kernel_evt = wrap_cl_error!(unsafe { exec.enqueue_nd_range(queue) }, "Failed to enqueue gemm kernel")?;
        c.set_dep(kernel_evt);
        Ok(())
    }
}
