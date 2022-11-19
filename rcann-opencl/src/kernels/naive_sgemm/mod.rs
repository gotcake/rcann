use std::rc::Rc;
use opencl3::command_queue::{cl_int, CommandQueue};
use opencl3::context::Context;
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::program::Program;
use rcann::tensor::{Dims, ITensor};
use crate::error::Error;
use crate::tensor::OclTensor;
use crate::util;
use crate::util::Result;

pub struct NaiveSGEMMKernel {
    program: Program,
    kernel: Kernel
}

impl NaiveSGEMMKernel {

    pub fn new(context: &Context) -> Result<NaiveSGEMMKernel> {
        let program = util::create_program(context, include_str!("naive_sgemm.cl"))?;
        let kernel = util::create_kernel(&program, "naive_sgemm")?;
        Ok(NaiveSGEMMKernel {
            program,
            kernel
        })
    }

    pub fn matrix_multiply(
        &self,
        queue: &CommandQueue,
        alpha: f32,
        a: &OclTensor<f32>,
        ta: bool,
        b: &OclTensor<f32>,
        tb: bool,
        beta: f32,
        c: &mut OclTensor<f32>,
        tc: bool,
    ) {
        assert_eq!(ta, false);
        assert_eq!(tb, false);
        assert_eq!(tc, false);

        let (m, k) = a.dims().unwrap_2d();
        let (_, n) = a.dims().unwrap_2d();

        assert_eq!(b.dims(), &Dims::D2(k, n));
        assert_eq!(c.dims(), &Dims::D2(m, n));

        let mut exec = ExecuteKernel::new(&self.kernel);

        unsafe {
            exec
                .set_arg(&(m as cl_int))// M
                .set_arg(&(k as cl_int)) // K
                .set_arg(&(n as cl_int)) // N
                .set_arg(&alpha) // ALPHA
                .set_arg(a.buffer()) // A
                .set_arg(b.buffer()) // B
                .set_arg(&beta) // BETA
                .set_arg(c.buffer_mut()) // C
        };

        assert_eq!(n % 32, 0);
        assert_eq!(m % 32, 0);
        exec.set_local_work_sizes(&[32, 32]) // TODO
            .set_global_work_sizes(&[m, n]);

        if let Some(evt) = a.get_dependency() {
            exec.set_wait_event(&evt);
        }
        if let Some(evt) = b.get_dependency() {
            exec.set_wait_event(&evt);
        }
        if let Some(evt) = c.get_dependency() {
            exec.set_wait_event(&evt);
        }

        let kernel_evt = unsafe {
            exec.enqueue_nd_range(queue)
                .expect("failed to enqueue kernel")
        };

        c.put_dependency(Rc::new(kernel_evt));

    }

}

#[cfg(test)]
mod test {
    use rand::distributions::Standard;
    use rand::prelude::StdRng;
    use rand::SeedableRng;
    use rcann::backend::{CpuBackend, MatrixMultiplication};
    use rcann::tensor::Tensor;
    use super::*;

    #[test]
    fn test_naive_sgemm() -> Result<()> {

        let cpu_backend = CpuBackend::<f32>::new();

        let device = util::get_default_device()?;
        let context = util::get_context(&device)?;
        let mut queue = util::create_queue(&context)?;
        let kernel = NaiveSGEMMKernel::new(&context)?;

        let mut rng = StdRng::seed_from_u64(0x1234567);

        let dims = Dims::D2(128, 128);

        let a_native = Tensor::from_distribution(&mut rng, Standard, &dims);
        let b_native = Tensor::from_distribution(&mut rng, Standard, &dims);
        let mut c_expected = Tensor::filled_default(&dims);

        cpu_backend.matmul(
            1.0,
            &a_native,
            false,
            &b_native,
            false,
            0.0,
            &mut c_expected,
            false
        );

        let a_ocl = OclTensor::from_native(&context, &queue, &a_native)?;
        let b_ocl = OclTensor::from_native(&context, &queue, &b_native)?;
        let mut c_ocl = OclTensor::new(&context, &dims)?;

        kernel.matrix_multiply(
            &queue,
            1.0,
            &a_ocl,
            false,
            &b_ocl,
            false,
            0.0,
            &mut c_ocl,
            false
        );

        let c_actual = c_ocl.as_native(&queue)?;

        println!("actual = {c_actual:?}");
        println!("expected = {c_expected:?}");

        assert_eq!(c_expected, c_actual);

        Ok(())
    }

}