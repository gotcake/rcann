pub mod naive_sgemm;

use crate::error::Error;
use crate::util;
use opencl3::context::Context;
use opencl3::kernel::Kernel;
use opencl3::memory::Buffer;
use opencl3::program::Program;
use rcann::dtype::DType;

struct OclTensor<T> {
    buffer: Buffer<T>,
}

trait OclKernel {
    type DType: DType;
}

#[derive(Debug)]
pub struct Kernels {
    sgemm_program: Program,
    pub sgemm: Kernel,
}

impl Kernels {
    pub fn create(context: &Context) -> Result<Self, Error> {
        let sgemm_program =
            util::create_program(context, include_str!("naive_sgemm/naive_sgemm.cl"))?;
        let sgemm = util::create_kernel(&sgemm_program, "naive_sgemm")?;
        Ok(Kernels {
            sgemm_program,
            sgemm,
        })
    }
}
