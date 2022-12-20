use approx::assert_abs_diff_eq;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::StandardNormal;
use rcann::backend::{BackendOther, CpuBackend};
use rcann::tensor::{Dim2, ITensor, Tensor2};
use crate::kernels::softmax::Softmax2;
use crate::tensor::{OclTensor2};
use crate::util::*;
use super::Softmax;

#[test]
fn test_softmax() {
    let TestContext { device, context, queue } = create_test_context().unwrap();
    let kernel = Softmax::create(&context, VecWidth::SIXTEEN).unwrap();
    let cpu = CpuBackend::<f32>::new(0);
    let mut rng = StdRng::seed_from_u64(0x93827291);

    let activation = Tensor2::from_distribution(&mut rng, StandardNormal, Dim2(20, 20));
    let mut output_expected = Tensor2::zeroed(activation.dims().clone());
    let activation_ocl = OclTensor2::from_native(&context, &queue, &activation).unwrap();
    let mut output_ocl = OclTensor2::zeroed(&context, &queue, activation.dims().clone()).unwrap();

    cpu.softmax(&activation, &mut output_expected);

    kernel.softmax(&queue, &activation_ocl, &mut output_ocl);

    assert_abs_diff_eq!(output_expected, output_ocl.as_native(&queue).unwrap(), epsilon = 0.001);

}

#[test]
fn test_softmax2() {
    let TestContext { device, context, queue } = create_test_context().unwrap();
    let cpu = CpuBackend::<f32>::new(0);
    let mut rng = StdRng::seed_from_u64(0x93827291);

    let activation = Tensor2::from_distribution(&mut rng, StandardNormal, Dim2(42, 50));
    let mut output_expected = Tensor2::zeroed(activation.dims().clone());
    let activation_ocl = OclTensor2::from_native(&context, &queue, &activation).unwrap();
    let mut output_ocl = OclTensor2::zeroed(&context, &queue, activation.dims().clone()).unwrap();

    cpu.softmax(&activation, &mut output_expected);

    let kernel = Softmax2::create(&context, FixedWidth2DProgramArgs {
        vec_width: VecWidth::SIXTEEN,
        cols: activation.dims().cols(),
        row_stride: activation_ocl.buffer_dims().cols(),
    }).unwrap();
    kernel.softmax(&queue, &activation_ocl, &mut output_ocl);

    assert_abs_diff_eq!(output_expected, output_ocl.as_native(&queue).unwrap(), epsilon = 0.001);

}