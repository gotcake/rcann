#[cfg(test)]
mod test;

use crate::tensor::event_list::EventList;
use crate::tensor::{OclFloat, OclTensor};
use crate::util::*;
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
    pub const VECTOR_WIDTH: usize = 16;
}

impl GemmKernel {
    pub fn new(context: &Context) -> Result<Self> {
        let mut code = format_c_defines!(
            "FLOAT_BITS" => 32,
            "TILE_SIZE" => constants::TILE_SIZE,
            "VECTOR_WIDTH" => constants::VECTOR_WIDTH,
        );
        code.push_str(include_str!("../types.cl"));
        code.push_str("\n");
        code.push_str(include_str!("gemm.cl"));
        let program = create_program(context, code.as_ref(), "")?;
        let kernel = create_kernel(&program, "gemm")?;
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
        exec.set_local_work_sizes(&[constants::TILE_SIZE, constants::TILE_SIZE / constants::VECTOR_WIDTH])
            .set_global_work_sizes(&[m, n / constants::VECTOR_WIDTH]);
        let deps = EventList::concat([a.deps(), b.deps(), c.deps()]);
        exec.set_event_wait_list(deps.as_slice());
        let kernel_evt = wrap_cl_error!(unsafe { exec.enqueue_nd_range(queue) }, "Failed to enqueue gemm kernel")?;
        c.set_deps(EventList::from_event(kernel_evt));
        Ok(())
    }
}

ocl_program! {
    name = GeMM,
    source = "gemm.cl",
    generic_args = <T: OclFloat>,
    compile_params = (
        vec_width: u8,
        tile_size: usize,
    ),
    validation = {
        validate!(is_valid_vec_width(*vec_width), "Invalid vector width");
        validate!(*tile_size % *vec_width as usize == 0, "tile_size must be a multiple of vec_width");
    },
    defines = {
        FLOAT_BITS = T::BITS,
        VECTOR_WIDTH = vec_width,
        TILE_SIZE = tile_size,
    },
    kernels = {
        gemm {
            call_params = (
                alpha: f32,
                a: &OclTensor<f32, Dim2>,
                b: &OclTensor<f32, Dim2>,
                beta: f32,
                c: &mut OclTensor<f32, Dim2>,
            ),
            pre = {
                let &Dim2(m, k) = a.buffer_dims();
                let &Dim2(_, n) = b.buffer_dims();
            },
            validation = {
                assert_eq!(b.buffer_dims(), &Dim2(k, n));
                assert_eq!(c.buffer_dims(), &Dim2(m, n));
                assert_eq!(m % *tile_size, 0);
                assert_eq!(n % *tile_size, 0);
                assert_eq!(k % *tile_size, 0);
            },
            inputs = [a, b, c],
            outputs = [c],
            kernel_args = [
                &(m as u32),
                &(k as u32),
                &(n as u32),
                &alpha,
                a.buffer(),
                b.buffer(),
                &beta,
                c.buffer(),
            ],
            global_dims = [m, n / *vec_width as usize],
            local_dims = [*tile_size, *tile_size / *vec_width as usize],
        },
    },
}
