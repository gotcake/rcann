use std::cmp::{max, min};
use crate::tensor::OclTensor;
use crate::util;
use crate::util::Result;
use opencl3::command_queue::{cl_int, CommandQueue};
use opencl3::context::Context;
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::program::Program;
use rcann::tensor::{Dim2, ITensor};
use std::rc::Rc;
use crate::util::MATRIX_PADDING_SIZE;
use crate::error::Error;

pub struct NaiveSGEMMKernel {
    program: Program,
    sgemm: Kernel,
    fill_pad_zero: Kernel,
}

impl NaiveSGEMMKernel {
    pub fn new(context: &Context) -> Result<NaiveSGEMMKernel> {
        let program = util::create_program(context, include_str!("naive_sgemm.cl"))?;
        let sgemm = util::create_kernel(&program, "naive_sgemm")?;
        let fill_pad_zero = util::create_kernel(&program, "fill_pad_zero")?;
        Ok(NaiveSGEMMKernel { program, sgemm, fill_pad_zero })
    }

    fn fill_pad_zero(&self, queue: &CommandQueue, tensor: &OclTensor<f32, Dim2>) -> Result<()> {
        let mut exec = ExecuteKernel::new(&self.fill_pad_zero);
        let &Dim2(rows, cols) = tensor.dims();
        let &Dim2(b_rows, b_cols) = tensor.buffer_dims();
        assert_eq!(b_rows % MATRIX_PADDING_SIZE, 0);
        assert_eq!(b_cols % MATRIX_PADDING_SIZE, 0);
        unsafe {
            exec.set_arg(&(rows as cl_int))
                .set_arg(&(cols as cl_int))
                .set_arg(&(b_rows as cl_int))
                .set_arg(&(b_cols as cl_int))
                .set_arg(tensor.buffer());
        }
        let n = max(b_cols, b_rows);
        exec.set_local_work_sizes(&[MATRIX_PADDING_SIZE])
            .set_global_work_sizes(&[n]);
        if let Some(evt) = tensor.get_dependency() {
            exec.set_wait_event(evt.as_ref());
        }
        let kernel_evt = unsafe {
            exec.enqueue_nd_range(queue)
                .map_err(|err| Error::from_cl_err(err, "Failed to enqueue fill_pad_zero kernel"))?
        };
        tensor.put_dependency(Rc::new(kernel_evt));
        Ok(())
    }

    fn matmul_impl(
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
        assert_eq!(m % MATRIX_PADDING_SIZE, 0);
        assert_eq!(n % MATRIX_PADDING_SIZE, 0);
        assert_eq!(k % MATRIX_PADDING_SIZE, 0);
        let mut exec = ExecuteKernel::new(&self.sgemm);
        unsafe {
            exec.set_arg(&(m as cl_int)) // M
                .set_arg(&(k as cl_int)) // K
                .set_arg(&(n as cl_int)) // N
                .set_arg(&alpha) // ALPHA
                .set_arg(a.buffer()) // A
                .set_arg(b.buffer()) // B
                .set_arg(&beta) // BETA
                .set_arg(c.buffer_mut()) // C
        };
        exec.set_local_work_sizes(&[MATRIX_PADDING_SIZE, MATRIX_PADDING_SIZE]) // TODO?
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
                .map_err(|err| Error::from_cl_err(err, "Failed to enqueue naive_sgemm kernel"))?
        };
        c.put_dependency(Rc::new(kernel_evt));
        Ok(())
    }

    pub fn matrix_multiply(
        &self,
        queue: &CommandQueue,
        alpha: f32,
        a: &OclTensor<f32, Dim2>,
        ta: bool,
        b: &OclTensor<f32, Dim2>,
        tb: bool,
        beta: f32,
        c: &mut OclTensor<f32, Dim2>,
        tc: bool,
    ) -> Result<()> {
        assert_eq!(ta, false);
        assert_eq!(tb, false);
        assert_eq!(tc, false);
        if a.dims() != a.buffer_dims() {
            self.fill_pad_zero(queue, &a)?;
        }
        if b.dims() != b.buffer_dims() {
            self.fill_pad_zero(queue, &b)?;
        }
        if beta != 0.0 && c.dims() != c.buffer_dims() {
            self.fill_pad_zero(queue, &c)?;
        }
        self.matmul_impl(
            queue,
            alpha,
            a,
            b,
            beta,
            c
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use approx::assert_abs_diff_eq;
    use super::*;
    use rand::prelude::StdRng;
    use rand::SeedableRng;
    use rand_distr::StandardNormal;
    use rcann::backend::{CpuBackend, MatrixMultiplication};
    use rcann::tensor::Tensor;

    #[test]
    fn test_naive_sgemm() -> Result<()> {
        let cpu_backend = CpuBackend::<f32>::new();

        let device = util::get_default_device()?;
        let context = util::get_context(&device)?;
        let mut queue = util::create_queue(&context)?;
        let kernel = NaiveSGEMMKernel::new(&context)?;

        let mut rng = StdRng::seed_from_u64(0x1234567);

        let m = 500;
        let k = 1000;
        let n = 200;

        let a_native = Tensor::from_distribution(&mut rng, StandardNormal, Dim2(m, k));
        let b_native = Tensor::from_distribution(&mut rng, StandardNormal, Dim2(k, n));
        let mut c_expected = Tensor::filled_default(Dim2(m, n));

        cpu_backend.matmul(
            1.0,
            &a_native,
            false,
            &b_native,
            false,
            0.0,
            &mut c_expected,
            false,
        );

        let a_ocl = OclTensor::from_native(&context, &queue, &a_native)?;
        let b_ocl = OclTensor::from_native(&context, &queue, &b_native)?;
        let mut c_ocl = OclTensor::new(&context, Dim2(m, n))?;

        kernel.matrix_multiply(
            &queue, 1.0, &a_ocl, false, &b_ocl, false, 0.0, &mut c_ocl, false,
        )?;

        let c_actual = c_ocl.as_native(&queue)?;

        println!("actual = {c_actual:?}");
        println!("expected = {c_expected:?}");

        assert_abs_diff_eq!(c_expected, c_actual, epsilon=0.001);

        Ok(())
    }
}
