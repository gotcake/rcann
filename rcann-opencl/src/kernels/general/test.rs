use crate::kernels::general::GeneralKernels;
use crate::tensor::{OclTensor1, OclTensor2};
use crate::util;
use crate::util::{Result, TestContext};
use approx::assert_abs_diff_eq;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::StandardNormal;
use rcann::backend::{BackendOther, CpuBackend};
use rcann::tensor::{Dim1, Dim2, ITensor, Tensor1, Tensor2};

#[test]
fn test_sigmoid() -> Result<()> {
    let TestContext { device, context, queue } = util::create_test_context()?;
    let kernel = GeneralKernels::new(&context)?;
    let cpu = CpuBackend::<f32>::new(0);
    let mut rng = StdRng::seed_from_u64(0x3827261);

    let input = Tensor2::from_distribution(&mut rng, StandardNormal, Dim2(30, 30));
    let mut expected = Tensor2::zeroed(*input.dims());
    cpu.sigmoid(&input, &mut expected);

    let input_ocl = OclTensor2::from_native(&context, &queue, &input)?;
    let mut output_ocl = OclTensor2::zeroed(&context, &queue, *input.dims())?;
    kernel.sigmoid(&queue, &input_ocl, &mut output_ocl)?;
    let actual = output_ocl.as_native(&queue)?;

    assert_abs_diff_eq!(expected, actual, epsilon = 0.001);

    Ok(())
}

#[test]
fn test_sigmoid_error() -> Result<()> {
    let TestContext { device, context, queue } = util::create_test_context()?;
    let kernel = GeneralKernels::new(&context)?;
    let cpu = CpuBackend::<f32>::new(0);
    let mut rng = StdRng::seed_from_u64(0x3827261);

    let output = Tensor2::from_distribution(&mut rng, StandardNormal, Dim2(30, 30));
    let error = Tensor2::from_distribution(&mut rng, StandardNormal, Dim2(30, 30));
    let mut expected = Tensor2::zeroed(*output.dims());
    cpu.sigmoid_error(&output, &error, &mut expected);

    let output_ocl = OclTensor2::from_native(&context, &queue, &output)?;
    let error_ocl = OclTensor2::from_native(&context, &queue, &error)?;
    let mut result_ocl = OclTensor2::zeroed(&context, &queue, *expected.dims())?;
    kernel.sigmoid_error(&queue, &output_ocl, &error_ocl, &mut result_ocl)?;
    let actual = result_ocl.as_native(&queue)?;

    assert_abs_diff_eq!(expected, actual, epsilon = 0.001);

    Ok(())
}

#[test]
fn test_add_assign() -> Result<()> {
    let TestContext { device, context, queue } = util::create_test_context()?;
    let kernel = GeneralKernels::new(&context)?;
    let cpu = CpuBackend::<f32>::new(0);
    let mut rng = StdRng::seed_from_u64(0x3827261);

    let input = Tensor2::from_distribution(&mut rng, StandardNormal, Dim2(10, 10));
    let output = Tensor2::from_distribution(&mut rng, StandardNormal, *input.dims());
    let mut expected = output.clone();
    cpu.add_assign(0.75, &input, 0.25, &mut expected);

    let input_ocl = OclTensor2::from_native(&context, &queue, &input)?;
    let mut output_ocl = OclTensor2::from_native(&context, &queue, &output)?;
    kernel.add_assign(&queue, 0.75, &input_ocl, 0.25, &mut output_ocl)?;
    let actual = output_ocl.as_native(&queue)?;

    assert_abs_diff_eq!(expected, actual, epsilon = 0.001);

    Ok(())
}

#[test]
fn test_column_sum() -> Result<()> {
    let TestContext { device, context, queue } = util::create_test_context()?;
    let kernel = GeneralKernels::new(&context)?;
    let cpu = CpuBackend::<f32>::new(0);
    let mut rng = StdRng::seed_from_u64(0x3827261);

    let input = Tensor2::from_distribution(&mut rng, StandardNormal, Dim2(30, 30));
    let output = Tensor1::from_distribution(&mut rng, StandardNormal, Dim1(input.dims().cols()));
    let mut expected = output.clone();
    cpu.column_sum(0.75, &input, 0.25, &mut expected);

    let input_ocl = OclTensor2::from_native(&context, &queue, &input)?;
    let mut output_ocl = OclTensor1::from_native(&context, &queue, &output)?;
    kernel.column_sum(&queue, 0.75, &input_ocl, 0.25, &mut output_ocl)?;
    let actual = output_ocl.as_native(&queue)?;

    assert_abs_diff_eq!(expected, actual, epsilon = 0.001);

    Ok(())
}
