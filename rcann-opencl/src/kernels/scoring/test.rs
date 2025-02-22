macro_rules! impl_tests {
    ($mod_name: ident, $ty:ty) => {
        mod $mod_name {

            use crate::kernels::scoring::ScoringProgram;
            use crate::tensor::OclTensor2;
            use crate::util;
            use crate::util::{Result, TestContext};
            use approx::assert_abs_diff_eq;
            use rand::rngs::StdRng;
            use rand::SeedableRng;
            use rand_distr::StandardNormal;
            use rcann::backend::{BackendOther, CpuBackend};
            use rcann::tensor::{Dim2, ITensor, Tensor2, TensorBase};

            #[test]
            fn test_accum_multiclass_confusion_matrix() -> Result<()> {
                let TestContext { device, context, queue } = util::create_test_context()?;
                let cpu = CpuBackend::<$ty>::new(0);
                let kernel = ScoringProgram::<$ty>::create(&context)?;
                let mut rng = StdRng::seed_from_u64(0x82379173);

                let output_vals = Tensor2::from_distribution(&mut rng, StandardNormal, Dim2(200, 10));
                let expected_vals = Tensor2::from_distribution(&mut rng, StandardNormal, *output_vals.dims());

                let mut matrix_expected = Tensor2::zeroed(Dim2(output_vals.dims().cols(), output_vals.dims().cols()));
                cpu.accum_confusion_matrix_multiclass(&mut matrix_expected, &output_vals, expected_vals.view());
                cpu.accum_confusion_matrix_multiclass(&mut matrix_expected, &output_vals, expected_vals.view());

                let ocl_output = OclTensor2::from_native(&context, &queue, &output_vals)?;
                let ocl_expected = OclTensor2::from_native(&context, &queue, &expected_vals)?;
                let mut matrix_actual = OclTensor2::zeroed(&context, &queue, *matrix_expected.dims())?;

                kernel.accum_multiclass_confusion_matrix(&context, &queue, &mut matrix_actual, &ocl_output, &ocl_expected);
                kernel.accum_multiclass_confusion_matrix(&context, &queue, &mut matrix_actual, &ocl_output, &ocl_expected);

                assert_abs_diff_eq!(matrix_expected, matrix_actual.as_native(&queue)?);

                Ok(())
            }
        }
    };
}

impl_tests!(tests_f32, f32);
impl_tests!(tests_f64, f64);
