use super::*;
use crate::util::TestContext;
use approx::assert_abs_diff_eq;
use rand::prelude::StdRng;
use rand::SeedableRng;
use rand_distr::StandardNormal;
use rcann::backend::{CpuBackend, MatrixMultiplication};
use rcann::tensor::{Tensor, TensorBase};

#[test]
fn test_gemm() -> Result<()> {
    let cpu_backend = CpuBackend::<f32>::new(0);

    let TestContext { device, context, queue } = util::create_test_context()?;
    let kernel = GemmKernel::new(&context)?;

    let mut rng = StdRng::seed_from_u64(0x1234567);

    let m = 50;
    let k = 20;
    let n = 30;

    let a_native = Tensor::from_distribution(&mut rng, StandardNormal, Dim2(m, k));
    let b_native = Tensor::from_distribution(&mut rng, StandardNormal, Dim2(k, n));
    let mut c_expected = Tensor::zeroed(Dim2(m, n));

    cpu_backend.matmul(1.0, a_native.view(), false, b_native.view(), false, 0.0, &mut c_expected);

    let a_ocl = OclTensor::from_native(&context, &queue, &a_native)?;
    let b_ocl = OclTensor::from_native(&context, &queue, &b_native)?;
    let mut c_ocl = OclTensor::zeroed(&context, &queue, Dim2(m, n))?;

    kernel.gemm(&queue, 1.0, &a_ocl, &b_ocl, 0.0, &mut c_ocl)?;

    let c_actual = c_ocl.as_native(&queue)?;

    //println!("actual = {c_actual:?}");
    //println!("expected = {c_expected:?}");

    assert_abs_diff_eq!(c_expected, c_actual, epsilon = 0.001);

    Ok(())
}
