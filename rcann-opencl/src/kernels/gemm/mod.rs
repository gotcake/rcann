#[cfg(test)]
mod test;

use crate::tensor::event_list::EventList;
use crate::tensor::{OclFloat, OclTensor};
use crate::util::*;
use opencl3::kernel::{ExecuteKernel};
use rcann::tensor::Dim2;
use crate::kernels::BUFFER_BLOCK_SIZE;

#[allow(unused)]
pub mod constants {
    pub const TILE_SIZE: usize = 16;
    pub const VECTOR_WIDTH: u8 = 16;
}

ocl_program! {
    name = GeMMProgram,
    source = "gemm.cl",
    generic_args = <T: OclFloat>,
    compile_params = (
        vec_width: VecWidth,
        tile_size: usize,
    ),
    validation = {
        validate!(is_power_of_two(*tile_size), "tile_size must be a power of two");
        validate!(*tile_size % *vec_width as usize == 0, "tile_size must be a multiple of vec_width");
        validate!(BUFFER_BLOCK_SIZE % *tile_size == 0, "BUFFER_BLOCK_SIZE must be a multiple of tile_size");
    },
    defines = {
        FLOAT_BITS = T::BITS,
        VECTOR_WIDTH = *vec_width,
        TILE_SIZE = *tile_size,
    },
    kernels = {
        gemm {
            call_params = (
                alpha: T,
                a: &OclTensor<T, Dim2>,
                b: &OclTensor<T, Dim2>,
                beta: T,
                c: &mut OclTensor<T, Dim2>,
            ),
            pre = {
                let m = a.buffer_dims().rows();
                let k = a.buffer_dims().cols();
                let n = b.buffer_dims().cols();
            },
            validation = {
                assert_eq!(b.buffer_dims().rows(), k);
                assert_eq!(c.buffer_dims(), &Dim2(m, n));
                assert_eq!(m % *tile_size, 0);
                assert_eq!(n % *tile_size, 0);
                assert_eq!(k % *tile_size, 0);
            },
            inputs = [a, b, c],
            outputs = [c],
            kernel_args = [
                &(m as u32),
                &(k as u32),
                &(n as u32),
                &alpha,
                a.buffer(),
                b.buffer(),
                &beta,
                c.buffer(),
            ],
            global_dims = [m, n / *vec_width as usize],
            local_dims = [*tile_size, *tile_size / *vec_width as usize],
        },
    },
}
