use bencher::{benchmark_group, benchmark_main, Bencher};
use rcann::backend::{BackendOther, TensorOps};
use rcann::util::bench::*;
use rcann_opencl::backend::OpenCLBackend;

macro_rules! impl_add_assign_bench {
    ($name:ident, $ty:ty, $factory:ident, $size:expr, $alpha:literal, $beta:literal) => {
        fn $name(bench: &mut Bencher) {
            let backend = OpenCLBackend::from_default_device(0).unwrap();
            let [a, b, _] = $factory($size);
            let ocl_a = backend.new_tensor_from_native(a);
            let mut ocl_b = backend.new_tensor_from_native(b);
            ocl_a.sync();
            ocl_b.sync();
            bench.iter(|| {
                backend.add_assign($alpha, &ocl_a, $beta, &mut ocl_b);
                ocl_b.sync();
            })
        }
    };
}

impl_add_assign_bench!(add_assign_lg, f32, get_square_matrices, SIZE_LG, 0.75, 0.25);

benchmark_group!(add_assign, add_assign_lg);

benchmark_main!(add_assign,);
