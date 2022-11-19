use super::math::{compute_jacobian_matrix, DTypeOps};
use crate::backend::{Backend, BackendOther, MatrixMultiplication, TensorOps, TensorTyped};
use crate::tensor::{Dims, ITensor, Tensor, TensorBase, TensorBaseMut};
use std::cell::RefCell;
use std::cmp::Ordering;
use std::fmt::{Debug, Formatter, Write};
use std::iter::zip;
use std::ops::{Deref, DerefMut};

pub struct CpuBackend<DT: DTypeOps> {
    temp_matrix: RefCell<Tensor<DT>>,
}

impl<DT: DTypeOps> CpuBackend<DT> {
    pub fn new() -> Self {
        CpuBackend {
            temp_matrix: RefCell::new(Tensor::empty()),
        }
    }
}

impl<DT: DTypeOps> TensorTyped for CpuBackend<DT> {
    type DType = DT;
    type Tensor = Tensor<DT>;
}

impl<DT: DTypeOps> TensorOps for CpuBackend<DT> {
    #[inline]
    fn new_tensor<D>(&self, dim: D) -> Self::Tensor where D: Into<Dims> {
        Tensor::filled_default(dim)
    }
    #[inline]
    fn resize_tensor<D>(&self, tensor: &mut Self::Tensor, dims: D) where D: Into<Dims> {
        tensor.resize_fill_default(dims)
    }
    fn write_tensor<T>(&self, tensor: &mut Self::Tensor, native_src: &T) where T: TensorBase<Self::DType> {
        assert_eq!(tensor.dims(), native_src.dims());
        tensor.as_mut().copy_from_slice(native_src.as_ref());
    }
    fn read_tensor<T>(&self, tensor: &Self::Tensor, native_dst: &mut T) where T: TensorBaseMut<Self::DType> {
        assert_eq!(tensor.dims(), native_dst.dims());
        native_dst.as_mut().copy_from_slice(tensor.as_ref());
    }
    #[inline]
    fn new_tensor_from_native<T>(&self, native: T) -> Self::Tensor where T: TensorBase<Self::DType> {
        native.into_owned()
    }
}

impl<DT: DTypeOps> MatrixMultiplication for CpuBackend<DT> {
    #[inline]
    fn matmul(&self, alpha: DT, a: &Self::Tensor, ta: bool, b: &Self::Tensor, tb: bool, beta: DT, c: &mut Self::Tensor, tc: bool) {
        DT::matrix_multiply(alpha, a, ta, b, tb, beta, c, tc);
    }
}

impl<DT: DTypeOps> BackendOther for CpuBackend<DT> {

    fn column_sum(
        &self,
        alpha: Self::DType,
        a: &Self::Tensor,
        beta: Self::DType,
        b: &mut Self::Tensor,
    ) {
        let (rows, cols) = a.dims().unwrap_2d();
        assert_eq!(b.dims(), &Dims::D1(cols));
        let pa = a.as_ptr();
        let pb = b.as_mut_ptr();
        let r = rows as isize;
        let c = cols as isize;
        for i in 0..c {
            let mut sum = DT::ZERO;
            for j in 0..r {
                sum += unsafe { *pa.offset(j * c + i) };
            }
            unsafe {
                let p = pb.offset(i);
                *p = sum * alpha + *p * beta;
            }
        }
    }

    // TODO: implement specialized versions of this
    fn add_assign(
        &self,
        alpha: Self::DType,
        a: &Self::Tensor,
        beta: Self::DType,
        b: &mut Self::Tensor,
    ) {
        assert_eq!(a.dims(), b.dims());
        for (&ai, bi) in zip(a, b) {
            *bi = alpha * ai + beta * *bi;
        }
    }

    fn sigmoid(&self, activation: &Self::Tensor, output: &mut Self::Tensor) {
        assert_eq!(activation.dims(), output.dims());
        for (o, &a) in zip(output, activation) {
            *o = DT::ONE / (DT::ONE + (-a).exp());
        }
    }

    fn sigmoid_error(
        &self,
        output: &Self::Tensor,
        out_error: &Self::Tensor,
        result: &mut Self::Tensor,
    ) {
        assert_eq!(output.dims(), result.dims());
        assert_eq!(output.dims(), out_error.dims());
        for ((r, &out), &err) in zip(zip(result, output), out_error) {
            *r = err * (out * (DT::ONE - out))
        }
    }

    fn relu(&self, leak: DT, activation: &Self::Tensor, output: &mut Self::Tensor) {
        assert_eq!(activation.dims(), output.dims());
        for (o, &a) in zip(output, activation) {
            *o = if a < DT::ZERO { a * leak } else { a }
        }
    }

    fn relu_error(
        &self,
        leak: DT,
        activation: &Self::Tensor,
        out_error: &Self::Tensor,
        result: &mut Self::Tensor,
    ) {
        assert_eq!(activation.dims(), result.dims());
        assert_eq!(activation.dims(), out_error.dims());
        for ((r, &act), &err) in zip(zip(result, activation), out_error) {
            *r = if act < DT::ZERO { leak * err } else { err };
        }
    }

    fn softmax(&self, activation: &Self::Tensor, output: &mut Self::Tensor) {
        assert_eq!(activation.dims().len(), 2);
        assert_eq!(activation.dims(), output.dims());
        for (mut output_row, activation_row) in
            zip(output.iter_first_axis_mut(), activation.iter_first_axis())
        {
            // shift the values by -max(inputs) to prevent overflow (does not affect derivative)
            let max = *activation_row
                .iter()
                .max_by(|&a, &b| {
                    if a > b {
                        Ordering::Greater
                    } else {
                        Ordering::Less
                    }
                })
                .unwrap();
            let mut sum = DT::ZERO;
            for (t, &a) in zip(output_row.iter_mut(), activation_row) {
                let x = (a - max).exp();
                sum += x;
                *t = x;
            }
            for t in output_row.iter_mut() {
                *t /= sum
            }
        }
    }

    fn softmax_error(
        &self,
        output: &Self::Tensor,
        out_error: &Self::Tensor,
        result: &mut Self::Tensor,
    ) {
        let (_, size) = output.dims().unwrap_2d();
        assert_eq!(output.dims(), result.dims());
        let mut temp = self.temp_matrix.borrow_mut();
        self.resize_tensor(temp.deref_mut(), (size, size));
        for (mut result_row, (output_row, out_err_row)) in zip(
            result.iter_first_axis_mut(),
            zip(output.iter_first_axis(), out_error.iter_first_axis()),
        ) {
            compute_jacobian_matrix(&output_row, temp.deref_mut());
            DT::matrix_multiply(
                DT::ONE,
                &out_err_row.as_row_matrix_2d(),
                false,
                temp.deref(),
                false,
                DT::ZERO,
                &mut result_row.as_row_matrix_2d_mut(),
                false,
            );
        }
    }

    fn mean_squared_error(
        &self,
        output: &Self::Tensor,
        expected: &Self::Tensor,
        result: &mut Self::Tensor,
        result_deriv: &mut Self::Tensor,
    ) {
        debug_assert_eq!(Some(output.dims().first()), result.dims().as_1d());
        debug_assert_eq!(output.dims(), expected.dims());
        debug_assert_eq!(output.dims(), result_deriv.dims());
        for (r, (rd_row, (o_row, e_row))) in zip(
            result,
            zip(
                result_deriv.iter_first_axis_mut(),
                zip(output.iter_first_axis(), expected.iter_first_axis()),
            ),
        ) {
            let mut sum_error = DT::ZERO;
            let count = DT::from_usize(rd_row.len());
            for (rd, (&o, &e)) in zip(rd_row, zip(o_row, e_row)) {
                let diff = o - e;
                *rd = diff;
                sum_error += diff * diff;
            }
            *r = sum_error / count;
        }
    }
}

impl<DT: DTypeOps> Backend for CpuBackend<DT> {}

impl<DT: DTypeOps> Debug for CpuBackend<DT> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("CpuBackend<")?;
        f.write_str(std::any::type_name::<DT>())?;
        f.write_char('>')
    }
}
