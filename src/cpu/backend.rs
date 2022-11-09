use std::cell::RefCell;
use std::cmp::Ordering;
use std::iter::zip;
use crate::backend::Backend;
use crate::cpu::math::{DTypeOps, jacobian};
use crate::cpu::matrix::{Matrix, MatrixBase, MatrixBaseMut, MatrixRef, MatrixRefMut, resize_vec};
use crate::tensor::Tensor2;

pub struct CpuBackend32 {
    temp_matrix: RefCell<Matrix<f32>>
}

impl CpuBackend32 {
    pub fn new() -> Self {
        CpuBackend32 {
            temp_matrix: RefCell::new(Matrix::empty())
        }
    }
}

impl Backend for CpuBackend32 {
    type DType = f32;
    type Tensor = Vec<f32>;
    type Tensor2 = Matrix<f32>;

    #[inline]
    fn new_tensor1(&self, len: usize) -> Self::Tensor {
        vec![0.0; len]
    }

    #[inline]
    fn new_tensor2(&self, rows: usize, cols: usize) -> Self::Tensor2 {
        Matrix::zero(rows, cols)
    }

    #[inline]
    fn new_tensor_from(&self, data: Vec<Self::DType>) -> Self::Tensor {
        data
    }

    #[inline]
    fn new_tensor2_from(&self, rows: usize, cols: usize, data: Vec<Self::DType>) -> Self::Tensor2 {
        Matrix::from_vec(rows, cols, data)
    }

    #[inline]
    fn resize_tensor1(&self, tensor: &mut Self::Tensor, new_len: usize) {
        resize_vec(tensor, new_len);
    }

    #[inline]
    fn resize_tensor2(&self, tensor: &mut Self::Tensor2, new_rows: usize, new_cols: usize) {
        tensor.resize(new_rows, new_cols);
    }

    #[inline]
    fn matmul(&self, alpha: f32, a: &Self::Tensor2, ta: bool, b: &Self::Tensor2, tb: bool, beta: f32, c: &mut Self::Tensor2, tc: bool) {
        f32::matmul(alpha, &a.view(), ta, &b.view(), tb, beta, &mut c.view_mut(), tc);
    }

    fn column_sum(&self, alpha: Self::DType, a: &Self::Tensor2, beta: Self::DType, b: &mut Self::Tensor) {
        assert_eq!(a.cols(), b.len());
        let pa = a.as_ptr();
        let pb = b.as_mut_ptr();
        let r = a.rows() as isize;
        let c = a.cols() as isize;
        for i in 0..c {
            let mut sum = 0.0;
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
    fn add_assign(&self, alpha: Self::DType, a: &Self::Tensor, beta: Self::DType, b: &mut Self::Tensor) {
        for (&ai, bi) in zip(a.iter(), b.iter_mut()) {
            *bi = alpha * ai + beta * *bi;
        }
    }
    #[inline]
    fn add_assign2(&self, alpha: Self::DType, a: &Self::Tensor2, beta: Self::DType, b: &mut Self::Tensor2) {
        for (&ai, bi) in zip(a.iter(), b.iter_mut()) {
            *bi = alpha * ai + beta * *bi;
        }
    }

    fn sigmoid(&self, activation: &Self::Tensor2, output: &mut Self::Tensor2) {
        for (o, &a) in zip(output.iter_mut(), activation.iter()) {
            *o = 1.0 / (1.0 + (-a).exp());
        }
    }

    fn sigmoid_error(&self, output: &Self::Tensor2, out_error: &Self::Tensor2, result: &mut Self::Tensor2) {
        for ((r, &out), &err) in zip(zip(result.iter_mut(), output.iter()), out_error.iter()) {
            *r = err * (out * (1.0 - out))
        }
    }

    fn relu(&self, leak: f32, activation: &Self::Tensor2, output: &mut Self::Tensor2) {
        for (o, &a) in zip(output.iter_mut(), activation.iter()) {
            *o = if a < 0.0 { a * leak } else { a }
        }
    }

    fn relu_error(&self, leak: f32, activation: &Self::Tensor2, out_error: &Self::Tensor2, result: &mut Self::Tensor2) {
        for ((r, &act), &err) in zip(zip(result.iter_mut(), activation.iter()), out_error.iter()) {
            *r = if act < 0.0 { leak * err } else { err };
        }
    }

    fn softmax(&self, activation: &Self::Tensor2, output: &mut Self::Tensor2) {
        for (t_arr, a_arr) in zip(output.iter_rows_mut(), activation.iter_rows()) {
            // shift the values by -max(inputs) to prevent overflow (does not affect derivative)
            let max = a_arr.iter().max_by(|a, b| if a > b { Ordering::Greater } else { Ordering::Less }).unwrap();
            let mut sum = 0.0;
            for (t, &a) in zip(t_arr.iter_mut(), a_arr) {
                let x = (a - max).exp();
                sum += x;
                *t = x;
            }
            for t in t_arr.iter_mut() {
                *t /= sum
            }
        }
    }

    fn softmax_error(&self, output: &Self::Tensor2, out_error: &Self::Tensor2, result: &mut Self::Tensor2) {
        let size = output.cols();
        let mut temp = self.temp_matrix.borrow_mut();
        temp.resize(size, size);
        for (result_row, (o_arr, oerr_arr)) in zip(result.iter_rows_mut(), zip(output.iter_rows(), out_error.iter_rows())) {
            jacobian(&o_arr, &mut temp.view_mut());
            f32::matmul(
                1.0,
                &MatrixRef::from_slice(1, size, oerr_arr),
                false,
                &temp.view(),
                false,
                0.0,
                &mut MatrixRefMut::from_slice(1, size, result_row),
                false
            );
        }
    }
}

