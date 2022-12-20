use crate::tensor::{OclTensor2, OclFloat};
use crate::util::*;
#[cfg(test)]
mod test;


ocl_program!(
    name = Softmax,
    source = "softmax.cl",
    generic_args = <T: OclFloat>,
    compile_param = vec_width: VecWidth,
    defines = {
        FLOAT_BITS = T::BITS,
        VEC_WIDTH = *vec_width as u8,
    },
    kernels = {
        softmax {
            call_params = (activation: &OclTensor2<T>, output: &mut OclTensor2<T>),
            inputs = [activation],
            outputs = [output],
            extra_args = [
                &(activation.dims().rows() as u32),
                &(activation.dims().cols() as u32),
                &(activation.buffer_dims().cols() as u32),
            ],
            global_dims = [activation.dims().rows()],
            local_dims = [],
            validation = {
                assert_eq!(activation.dims(), output.dims());
                assert_eq!(activation.buffer_dims(), output.buffer_dims());
                assert_eq!(activation.buffer_dims().cols() % *vec_width as usize, 0);
            },
        },
    },
);


row_based_ocl_program!(
    name = Softmax2,
    source = "softmax2.cl",
    kernels = {
        softmax {
            call_params = (activation: &OclTensor2<T>, output: &mut OclTensor2<T>),
            inputs = [activation],
            outputs = [output],
            global_dims = [activation.dims().rows()],
        },
    },
);