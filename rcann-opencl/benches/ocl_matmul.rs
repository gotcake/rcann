#[macro_use]
extern crate bencher;

use bencher::Bencher;
use rcann::backend::{MatrixMultiplication, TensorOps};
use rcann::util::bench::*;
use rcann_opencl::backend::OpenCLBackend;

macro_rules! impl_bench {
    ($name:ident, $ty:ty, $factory:ident, $size:expr, $alpha:literal, $ta:literal, $tb:literal, $beta:literal, $tc:literal) => {
        fn $name(bench: &mut Bencher) {
            let backend = OpenCLBackend::from_default_device().unwrap();
            let [a, b, c] = $factory($size);
            let ocl_a = backend.new_tensor_from_native(a);
            let ocl_b = backend.new_tensor_from_native(b);
            let mut ocl_c = backend.new_tensor_from_native(c);
            ocl_a.sync().unwrap();
            ocl_b.sync().unwrap();
            ocl_c.sync().unwrap();
            bench.iter(|| {
                backend.matmul(
                    $alpha,
                    &ocl_a,
                    $ta,
                    &ocl_b,
                    $tb,
                    0.0,
                    &mut ocl_c,
                    $tc
                );
                ocl_c.sync().unwrap();
            })
        }
    };
}

impl_bench!(ocl_f32_lg, f32, get_square_matmul_tensors, MATRIX_SIZE_LG, 1.0, false, false, 0.0, false);
impl_bench!(ocl_f32_md, f32, get_square_matmul_tensors, MATRIX_SIZE_MD, 1.0, false, false, 0.0, false);
impl_bench!(ocl_f32_sm, f32, get_square_matmul_tensors, MATRIX_SIZE_SM, 1.0, false, false, 0.0, false);

impl_bench!(ocl_f32_lg_transpose_a, f32, get_square_matmul_tensors, MATRIX_SIZE_LG, 1.0, true, false, 0.0, false);
impl_bench!(ocl_f32_md_transpose_a, f32, get_square_matmul_tensors, MATRIX_SIZE_MD, 1.0, true, false, 0.0, false);
impl_bench!(ocl_f32_sm_transpose_a, f32, get_square_matmul_tensors, MATRIX_SIZE_SM, 1.0, true, false, 0.0, false);

impl_bench!(ocl_f32_lg_transpose_b, f32, get_square_matmul_tensors, MATRIX_SIZE_LG, 1.0, false, true, 0.0, false);
impl_bench!(ocl_f32_md_transpose_b, f32, get_square_matmul_tensors, MATRIX_SIZE_MD, 1.0, false, true, 0.0, false);
impl_bench!(ocl_f32_sm_transpose_b, f32, get_square_matmul_tensors, MATRIX_SIZE_SM, 1.0, false, true, 0.0, false);

benchmark_group!(ocl_f32, ocl_f32_lg, ocl_f32_md, ocl_f32_sm);
benchmark_group!(ocl_f32_transpose_a, ocl_f32_lg_transpose_a, ocl_f32_md_transpose_a, ocl_f32_sm_transpose_a);
benchmark_group!(ocl_f32_transpose_b, ocl_f32_lg_transpose_b, ocl_f32_md_transpose_b, ocl_f32_sm_transpose_b);

benchmark_main!(ocl_f32, ocl_f32_transpose_a, ocl_f32_transpose_b);
