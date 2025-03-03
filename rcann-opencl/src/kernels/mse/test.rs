macro_rules! impl_tests {
    ($mod_name:ident, $ty:ty) => {
        mod $mod_name {

            use crate::kernels::mse::MSEProgram;
            use crate::tensor::{OclTensor1, OclTensor2};
            use crate::util;
            use crate::util::{Result, TestContext, VecWidth};
            use approx::assert_abs_diff_eq;
            use rand::prelude::StdRng;
            use rand::SeedableRng;
            use rand_distr::StandardNormal;
            use rcann::backend::{BackendOther, CpuBackend};
            use rcann::tensor::{Dim1, Dim2, ITensor, Tensor1, Tensor2, TensorBase};
            use std::num::NonZeroUsize;

            #[test]
            fn test_mean_squared_error() -> Result<()> {
                let TestContext { device, context, queue } = util::create_test_context()?;
                let cpu = CpuBackend::<$ty>::new(0);
                let mut rng = StdRng::seed_from_u64(0x82379173);

                let output_vals = Tensor2::from_distribution(&mut rng, StandardNormal, Dim2(90, 80));
                let expected_vals = Tensor2::from_distribution(&mut rng, StandardNormal, *output_vals.dims());

                let mut expected_result = Tensor1::zeroed(Dim1(output_vals.dims().rows()));
                let mut expected_result_deriv = Tensor2::zeroed(*output_vals.dims());
                cpu.mean_squared_error(
                    &output_vals,
                    expected_vals.view(),
                    &mut expected_result,
                    &mut expected_result_deriv,
                );

                let output_vals_ocl = OclTensor2::from_native(&context, &queue, &output_vals)?;
                let expected_vals_ocl = OclTensor2::from_native(&context, &queue, &expected_vals)?;
                let mut actual_result = OclTensor1::zeroed(&context, &queue, *expected_result.dims())?;
                let mut actual_result_deriv = OclTensor2::zeroed(&context, &queue, *expected_result_deriv.dims())?;

                let kernel = MSEProgram::<$ty>::create(
                    &context,
                    VecWidth::SIXTEEN,
                    output_vals_ocl.dims().cols(),
                    output_vals_ocl.buffer_dims().cols(),
                )?;
                kernel.mean_squared_error(
                    &queue,
                    &output_vals_ocl,
                    &expected_vals_ocl,
                    &mut actual_result,
                    &mut actual_result_deriv,
                );

                assert_abs_diff_eq!(expected_result, actual_result.as_native(&queue)?, epsilon = 0.001);
                assert_abs_diff_eq!(
                    expected_result_deriv,
                    actual_result_deriv.as_native(&queue)?,
                    epsilon = 0.001
                );

                Ok(())
            }
        }
    };
}

impl_tests!(tests_f32, f32);
impl_tests!(tests_f64, f64);
