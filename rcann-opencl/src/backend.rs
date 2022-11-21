use crate::kernels::gemm::GemmKernel;
use crate::kernels::transpose::TransposeKernel;
use crate::kernels::zero_padding::ZeroPaddingKernel;
use crate::tensor::OclTensor;
use crate::util::{self, panic_on_error, Result};
use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::Device;
use opencl3::types::cl_float;
use rcann::backend::{MatrixMultiplication, TensorOps, TensorTyped};
use rcann::dtype::DType;
use rcann::tensor::{Dim2, Dims, ITensor, TensorBase, TensorBaseMut};
use std::fmt::Debug;

#[derive(Debug)]
#[allow(unused)]
pub struct OpenCLBackend {
    device: Device,
    context: Context,
    queue: CommandQueue, // RefCell to grant mutability with immutable self,
    gemm_kernel: GemmKernel,
    transpose_kernel: TransposeKernel,
    zero_padding_kernel: ZeroPaddingKernel,
}

impl OpenCLBackend {
    pub fn from_default_device() -> Result<Self> {
        Self::from_device(util::get_default_device()?)
    }

    pub fn from_device(device: Device) -> Result<Self> {
        let context = util::get_context(&device)?;
        let queue = util::create_queue(&context)?;
        let gemm_kernel = GemmKernel::new(&context)?;
        let transpose_kernel = TransposeKernel::create(&context)?;
        let zero_padding_kernel = ZeroPaddingKernel::create(&context)?;
        Ok(OpenCLBackend {
            device,
            context,
            queue,
            gemm_kernel,
            transpose_kernel,
            zero_padding_kernel,
        })
    }
    #[inline]
    pub fn device(&self) -> &Device {
        &self.device
    }
    #[inline]
    pub fn context(&self) -> &Context {
        &self.context
    }
    #[inline]
    pub fn queue(&self) -> &CommandQueue {
        &self.queue
    }
}

impl TensorTyped for OpenCLBackend {
    type DType = cl_float;
    type Tensor<D: Dims> = OclTensor<cl_float, D>;
}

impl TensorOps for OpenCLBackend {
    #[inline]
    fn new_tensor<D: Dims>(&self, dim: D) -> Self::Tensor<D> {
        OclTensor::zeroed(&self.context, &self.queue, dim).unwrap()
    }

    fn resize_tensor<D: Dims>(&self, tensor: &mut Self::Tensor<D>, dims: D) {
        if &dims != tensor.dims() {
            panic!("OpenCLBackend does not support Tensor resizing");
        }
    }

    fn write_tensor<T, D>(&self, tensor: &mut Self::Tensor<D>, native_src: &T)
    where
        T: TensorBase<Self::DType, D>,
        D: Dims,
    {
        tensor.write_sync(&self.queue, native_src).unwrap();
    }

    fn read_tensor<T, D>(&self, tensor: &Self::Tensor<D>, native_dst: &mut T)
    where
        T: TensorBaseMut<Self::DType, D>,
        D: Dims,
    {
        tensor.read_sync(&self.queue, native_dst).unwrap();
    }
}

impl MatrixMultiplication for OpenCLBackend {
    fn matmul(
        &self,
        alpha: Self::DType,
        a: &Self::Tensor<Dim2>,
        ta: bool,
        b: &Self::Tensor<Dim2>,
        tb: bool,
        beta: Self::DType,
        c: &mut Self::Tensor<Dim2>,
        tc: bool,
    ) {
        panic_on_error(|| {
            let a_transpose = if ta {
                let mut temp = unsafe { OclTensor::uninit(&self.context, a.dims().transposed())? };
                self.transpose_kernel.transpose(&self.queue, a, &mut temp)?;
                Some(temp)
            } else {
                None
            };
            let b_transpose = if tb {
                let mut temp = unsafe { OclTensor::uninit(&self.context, b.dims().transposed())? };
                self.transpose_kernel.transpose(&self.queue, b, &mut temp)?;
                Some(temp)
            } else {
                None
            };
            let mut c_transpose = if tc {
                let mut temp = unsafe { OclTensor::uninit(&self.context, c.dims().transposed())? };
                if beta != Self::DType::ZERO {
                    self.transpose_kernel.transpose(&self.queue, c, &mut temp)?;
                }
                Some(temp)
            } else {
                None
            };
            self.gemm_kernel.gemm(
                &self.queue,
                alpha,
                a_transpose.as_ref().unwrap_or(a),
                b_transpose.as_ref().unwrap_or(b),
                beta,
                c_transpose.as_mut().unwrap_or(c),
            )?;
            if let Some(c_transpose) = c_transpose {
                self.transpose_kernel.transpose(&self.queue, &c_transpose, c)?;
            }
            Ok(())
        });
    }
}

#[cfg(test)]
mod test {
    use crate::backend::OpenCLBackend;
    use crate::util::Result;
    use approx::assert_abs_diff_eq;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rand_distr::StandardNormal;
    use rcann::backend::{CpuBackend, MatrixMultiplication, TensorOps};
    use rcann::tensor::{Dim2, Tensor2};

    macro_rules! impl_test_matrix_multiply {
        ($name:ident, $ty:ty, $m:literal, $k:literal, $n:literal, $alpha:literal, $beta:literal, $ta:literal, $tb:literal, $tc:literal, $seed:literal) => {
            #[test]
            fn $name() -> Result<()> {
                let mut rng = StdRng::seed_from_u64($seed);
                let cpu = CpuBackend::<$ty>::new();
                let ocl = OpenCLBackend::from_default_device()?;

                let dim_a = if $ta { Dim2($k, $m) } else { Dim2($m, $k) };
                let dim_b = if $tb { Dim2($n, $k) } else { Dim2($k, $n) };
                let dim_c = if $tc { Dim2($n, $m) } else { Dim2($m, $n) };

                let a = Tensor2::<$ty>::from_distribution(&mut rng, StandardNormal, dim_a);
                let b = Tensor2::<$ty>::from_distribution(&mut rng, StandardNormal, dim_b);
                let c = Tensor2::<$ty>::from_distribution(&mut rng, StandardNormal, dim_c);

                let mut c_expected = c.clone();
                cpu.matmul(
                    $alpha as $ty,
                    &a,
                    $ta,
                    &b,
                    $tb,
                    $beta as $ty,
                    &mut c_expected,
                    $tc,
                );

                let ocl_a = ocl.new_tensor_from_native(a);
                let ocl_b = ocl.new_tensor_from_native(b);
                let mut ocl_c = ocl.new_tensor_from_native(c);

                ocl.matmul(
                    $alpha as $ty,
                    &ocl_a,
                    $ta,
                    &ocl_b,
                    $tb,
                    $beta as $ty,
                    &mut ocl_c,
                    $tc,
                );

                let c_actual = ocl_c.as_native(ocl.queue())?;

                assert_abs_diff_eq!(c_expected, c_actual, epsilon = 0.001);

                Ok(())
            }
        };
    }

    impl_test_matrix_multiply!(test_small, f32, 2, 3, 2, 1, 0, false, false, false, 0x1234567);
    impl_test_matrix_multiply!(test_small_beta, f32, 2, 3, 2, 1, 1, false, false, false, 0x1234567);
    impl_test_matrix_multiply!(test_small_t_a, f32, 2, 3, 2, 1, 0, true, false, false, 0x1234567);
    impl_test_matrix_multiply!(test_small_t_b, f32, 2, 3, 2, 1, 0, false, true, false, 0x1234567);
    impl_test_matrix_multiply!(test_small_t_c, f32, 2, 3, 2, 1, 0, false, false, true, 0x1234567);
    impl_test_matrix_multiply!(test_small_t_c_beta, f32, 2, 3, 2, 1, 1, false, false, true, 0x1234567);
    impl_test_matrix_multiply!(test_small_t_ab, f32, 2, 3, 2, 1, 0, true, true, false, 0x1234567);
    impl_test_matrix_multiply!(test_small_t_abc, f32, 2, 3, 2, 1, 0, true, true, true, 0x1234567);
    impl_test_matrix_multiply!(test_small_t_abc_beta, f32, 2, 3, 2, 1, 1, true, true, true, 0x1234567);

    impl_test_matrix_multiply!(test_1block, f32, 16, 16, 16, 1, 0, false, false, false, 0x1234567);
    impl_test_matrix_multiply!(test_1block_t_a, f32, 16, 16, 16, 1, 0, true, false, false, 0x1234567);
    impl_test_matrix_multiply!(test_1block_t_b, f32, 16, 16, 16, 1, 0, false, true, false, 0x1234567);
    impl_test_matrix_multiply!(
        test_1block_t_c_beta,
        f32,
        16,
        16,
        16,
        1,
        1,
        false,
        false,
        true,
        0x1234567
    );
    impl_test_matrix_multiply!(test_1block_t_ab, f32, 16, 16, 16, 1, 0, true, true, false, 0x1234567);
    impl_test_matrix_multiply!(test_1block_t_abc, f32, 16, 16, 16, 1, 0, true, true, true, 0x1234567);

    impl_test_matrix_multiply!(test_med, f32, 123, 75, 203, 0.75, 0.25, false, false, false, 0x1234567);
    impl_test_matrix_multiply!(test_med_t, f32, 123, 75, 203, 0.75, 0.25, true, true, true, 0x1234567);

    impl_test_matrix_multiply!(test_large, f32, 1024, 512, 1024, 0.75, 0.25, false, false, false, 0x1234567);
    impl_test_matrix_multiply!(
        test_large_t,
        f32,
        1024,
        512,
        1024,
        0.75,
        0.25,
        true,
        true,
        true,
        0x1234567
    );
}
