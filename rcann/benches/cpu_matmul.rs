#[macro_use]
extern crate bencher;

use bencher::Bencher;
use rcann::backend::{CpuBackend, MatrixMultiplication};
use rcann::util::bench::*;

macro_rules! impl_bench {
    ($name:ident, $ty:ty, $factory:ident, $size:expr, $alpha:literal, $ta:literal, $tb:literal, $beta:literal, $tc:literal) => {
        fn $name(bench: &mut Bencher) {
            let backend = CpuBackend::<$ty>::new();
            let [a, b, mut c] = $factory($size);
            bench.iter(|| backend.matmul($alpha, &a, $ta, &b, $tb, $beta, &mut c, $tc))
        }
    };
}

impl_bench!(
    cpu_f32_lg,
    f32,
    get_square_matrices,
    SIZE_LG,
    1.0,
    false,
    false,
    0.0,
    false
);
impl_bench!(
    cpu_f32_md,
    f32,
    get_square_matrices,
    SIZE_MD,
    1.0,
    false,
    false,
    0.0,
    false
);
impl_bench!(
    cpu_f32_sm,
    f32,
    get_square_matrices,
    SIZE_SM,
    1.0,
    false,
    false,
    0.0,
    false
);
benchmark_group!(cpu_f32, cpu_f32_lg, cpu_f32_md, cpu_f32_sm);

impl_bench!(
    cpu_f64_lg,
    f64,
    get_square_matrices,
    SIZE_LG,
    1.0,
    false,
    false,
    0.0,
    false
);
impl_bench!(
    cpu_f64_md,
    f64,
    get_square_matrices,
    SIZE_MD,
    1.0,
    false,
    false,
    0.0,
    false
);
impl_bench!(
    cpu_f64_sm,
    f64,
    get_square_matrices,
    SIZE_SM,
    1.0,
    false,
    false,
    0.0,
    false
);
benchmark_group!(cpu_f64, cpu_f64_lg, cpu_f64_md, cpu_f64_sm);

/*
impl_bench!(cpu_f32_lg_transpose_a, f32, get_square_matmul_tensors, MATRIX_SIZE_LG, 1.0, true, false, 0.0, false);
impl_bench!(cpu_f32_md_transpose_a, f32, get_square_matmul_tensors, MATRIX_SIZE_MD, 1.0, true, false, 0.0, false);
impl_bench!(cpu_f32_sm_transpose_a, f32, get_square_matmul_tensors, MATRIX_SIZE_SM, 1.0, true, false, 0.0, false);
benchmark_group!(cpu_f32_transpose_a, cpu_f32_lg_transpose_a, cpu_f32_md_transpose_a, cpu_f32_sm_transpose_a);

impl_bench!(cpu_f32_lg_transpose_b, f32, get_square_matmul_tensors, MATRIX_SIZE_LG, 1.0, false, true, 0.0, false);
impl_bench!(cpu_f32_md_transpose_b, f32, get_square_matmul_tensors, MATRIX_SIZE_MD, 1.0, false, true, 0.0, false);
impl_bench!(cpu_f32_sm_transpose_b, f32, get_square_matmul_tensors, MATRIX_SIZE_SM, 1.0, false, true, 0.0, false);
benchmark_group!(cpu_f32_transpose_b, cpu_f32_lg_transpose_b, cpu_f32_md_transpose_b, cpu_f32_sm_transpose_b);
 */

benchmark_main!(
    cpu_f32, cpu_f64,
    //cpu_f32_transpose_a,
    //cpu_f32_transpose_b
);
