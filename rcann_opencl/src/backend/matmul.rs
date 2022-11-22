use crate::backend::OpenCLBackend;
use crate::tensor::OclTensor;
use crate::util::panic_on_error;
use rcann::backend::MatrixMultiplication;
use rcann::dtype::DType;
use rcann::tensor::{Dim2, ITensor};

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
    ) {
        panic_on_error(|| {
            let a_transpose = if ta {
                let mut temp = unsafe { OclTensor::uninit(&self.context, a.dims().transposed())? };
                self.transpose_kernel.transpose(&self.queue, a, &mut temp)?;
                Some(temp)
            } else {
                self.zero_padding_kernel.zero_padding(&self.queue, a)?;
                None
            };
            let b_transpose = if tb {
                let mut temp = unsafe { OclTensor::uninit(&self.context, b.dims().transposed())? };
                self.transpose_kernel.transpose(&self.queue, b, &mut temp)?;
                Some(temp)
            } else {
                self.zero_padding_kernel.zero_padding(&self.queue, b)?;
                None
            };
            if beta != Self::DType::ZERO {
                self.zero_padding_kernel.zero_padding(&self.queue, c)?;
            }
            self.gemm_kernel.gemm(
                &self.queue,
                alpha,
                a_transpose.as_ref().unwrap_or(a),
                b_transpose.as_ref().unwrap_or(b),
                beta,
                c,
            )?;
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
        ($name:ident, $ty:ty, $m:literal, $k:literal, $n:literal, $alpha:literal, $beta:literal, $ta:literal, $tb:literal, $seed:literal) => {
            #[test]
            fn $name() -> Result<()> {
                let mut rng = StdRng::seed_from_u64($seed);
                let cpu = CpuBackend::<$ty>::new(0);
                let ocl = OpenCLBackend::from_default_device(0)?;

                let dim_a = if $ta { Dim2($k, $m) } else { Dim2($m, $k) };
                let dim_b = if $tb { Dim2($n, $k) } else { Dim2($k, $n) };
                let dim_c = Dim2($m, $n);

                let a = Tensor2::<$ty>::from_distribution(&mut rng, StandardNormal, dim_a);
                let b = Tensor2::<$ty>::from_distribution(&mut rng, StandardNormal, dim_b);
                let c = Tensor2::<$ty>::from_distribution(&mut rng, StandardNormal, dim_c);

                let mut c_expected = c.clone();
                cpu.matmul($alpha as $ty, &a, $ta, &b, $tb, $beta as $ty, &mut c_expected);

                let ocl_a = ocl.new_tensor_from_native(a);
                let ocl_b = ocl.new_tensor_from_native(b);
                let mut ocl_c = ocl.new_tensor_from_native(c);

                ocl.matmul($alpha as $ty, &ocl_a, $ta, &ocl_b, $tb, $beta as $ty, &mut ocl_c);

                let c_actual = ocl_c.as_native(ocl.queue())?;

                assert_abs_diff_eq!(c_expected, c_actual, epsilon = 0.001);

                Ok(())
            }
        };
    }

    impl_test_matrix_multiply!(test_small, f32, 2, 3, 2, 1, 0, false, false, 0x1234567);
    impl_test_matrix_multiply!(test_small_beta, f32, 2, 3, 2, 1, 1, false, false, 0x1234567);
    impl_test_matrix_multiply!(test_small_t_a, f32, 2, 3, 2, 1, 0, true, false, 0x1234567);
    impl_test_matrix_multiply!(test_small_t_b, f32, 2, 3, 2, 1, 0, false, true, 0x1234567);
    impl_test_matrix_multiply!(test_small_t_ab, f32, 2, 3, 2, 1, 0, true, true, 0x1234567);

    impl_test_matrix_multiply!(test_1block, f32, 16, 16, 16, 1, 0, false, false, 0x1234567);
    impl_test_matrix_multiply!(test_1block_t_a, f32, 16, 16, 16, 1, 0, true, false, 0x1234567);
    impl_test_matrix_multiply!(test_1block_t_b, f32, 16, 16, 16, 1, 0, false, true, 0x1234567);

    impl_test_matrix_multiply!(test_1block_t_ab, f32, 16, 16, 16, 1, 0, true, true, 0x1234567);

    impl_test_matrix_multiply!(test_med, f32, 123, 75, 203, 0.75, 0.25, false, false, 0x1234567);

    impl_test_matrix_multiply!(test_large, f32, 1024, 512, 1024, 0.75, 0.25, false, false, 0x1234567);
    impl_test_matrix_multiply!(test_large_t, f32, 1024, 512, 1024, 0.75, 0.25, true, true, 0x1234567);
}
