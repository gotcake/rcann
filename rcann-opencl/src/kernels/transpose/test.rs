use approx::{assert_abs_diff_eq};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::StandardNormal;
use rcann::tensor;
use rcann::tensor::{Dim2, ITensor, Tensor};
use crate::kernels::transpose::TransposeKernel;
use crate::tensor::OclTensor;
use crate::util::{self, Result, TestContext};

#[test]
fn test_transpose_small_square() -> Result<()> {
    let TestContext { device, context, queue} = util::create_test_context()?;
    let kernel = TransposeKernel::create(&context)?;

    let input = tensor![[1., 2., 3.],[4., 5., 6.],[7., 8., 9.]];
    let input_ocl = OclTensor::from_native(&context, &queue, &input)?;
    let mut output_ocl = OclTensor::zeroed(&context, &queue, input.dims().transposed())?;
    kernel.transpose(&queue, &input_ocl, &mut output_ocl)?;
    let output = output_ocl.as_native(&queue)?;
    assert_eq!(output, tensor![[1., 4., 7.], [2., 5., 8.], [3., 6., 9.]]);

    Ok(())
}

#[test]
fn test_transpose_small_rectangle() -> Result<()> {
    let TestContext { device, context, queue} = util::create_test_context()?;
    let kernel = TransposeKernel::create(&context)?;

    let input = tensor![[1., 2., 3.],[4., 5., 6.]];
    let input_ocl = OclTensor::from_native(&context, &queue, &input)?;
    let mut output_ocl = OclTensor::zeroed(&context, &queue, input.dims().transposed())?;
    kernel.transpose(&queue, &input_ocl, &mut output_ocl)?;
    let output = output_ocl.as_native(&queue)?;
    assert_eq!(output, tensor![[1., 4.], [2., 5.], [3., 6.]]);

    Ok(())
}

#[test]
fn test_transpose_large() -> Result<()> {

    let TestContext { device, context, queue} = util::create_test_context()?;
    let kernel = TransposeKernel::create(&context)?;
    let mut rng = StdRng::seed_from_u64(0x1234567);

    let n = 200;
    let m = 300;
    let input = Tensor::from_distribution(&mut rng, StandardNormal, Dim2(m, n));
    let mut input_ocl = OclTensor::from_native(&context, &queue, &input)?;
    let mut output_ocl = OclTensor::zeroed(&context, &queue, input.dims().transposed())?;
    kernel.transpose(&queue, &input_ocl, &mut output_ocl)?;
    kernel.transpose(&queue, &output_ocl, &mut input_ocl)?;
    let output = input_ocl.as_native(&queue)?;
    assert_abs_diff_eq!(output, input);

    Ok(())
}