use super::math::{compute_jacobian_matrix, DTypeOps};
use crate::backend::{Backend, BackendOther, MatrixMultiplication, TensorOps, TensorTyped};
use crate::tensor::{Dim2, Dims, DimsMore, DimsZero, ITensor, Tensor, Tensor1, Tensor2, TensorBase, TensorBaseMut, TensorView, TensorView2};
use std::cell::RefCell;
use std::cmp::Ordering;
use std::fmt::{Debug, Formatter, Write};
use std::iter::zip;
use std::ops::{Deref, DerefMut};
use crate::backend::cpu::math::argmax;

pub struct CpuBackend<DT: DTypeOps> {
    max_batch_size: usize,
    temp_matrix: RefCell<Tensor2<DT>>,
}

impl<DT: DTypeOps> CpuBackend<DT> {
    pub fn new(max_batch_size: usize) -> Self {
        CpuBackend {
            max_batch_size,
            temp_matrix: RefCell::new(Tensor2::empty()),
        }
    }
}

impl<DT: DTypeOps> TensorTyped for CpuBackend<DT> {
    type Float = DT;
    type TensorRef<'a, D: Dims> = TensorView<'a, DT, D>;
    type Tensor<D: Dims> = Tensor<DT, D>;
    type InputAdaptionBuff<D: Dims> = ();
    type OutputAdaptionBuff<D: Dims> = ();
}

impl<DT: DTypeOps> TensorOps for CpuBackend<DT> {
    #[inline]
    fn new_tensor_exact<D: Dims>(&self, dim: D) -> Tensor<DT, D> {
        Tensor::zeroed(dim)
    }

    fn new_tensor_batch_sized<D: DimsMore>(&self, inner_dims: D) -> Tensor<DT, D::More> {
        Tensor::zeroed(inner_dims.insert_major(self.max_batch_size))
    }

    #[inline]
    fn resize_tensor<D: Dims>(&self, tensor: &mut Tensor<DT, D>, dims: D) {
        tensor.resize_within_capacity(DT::ZERO, dims)
    }

    fn write_tensor<T, D>(&self, tensor: &mut Tensor<DT, D>, native_src: &T)
    where
        T: TensorBase<Self::Float, D>,
        D: Dims,
    {
        assert_eq!(tensor.dims(), native_src.dims());
        tensor.as_mut().copy_from_slice(native_src.as_ref());
    }
    fn read_tensor<T, D>(&self, tensor: &Tensor<DT, D>, native_dst: &mut T)
    where
        T: TensorBaseMut<Self::Float, D>,
        D: Dims,
    {
        assert_eq!(tensor.dims(), native_dst.dims());
        native_dst.as_mut().copy_from_slice(tensor.as_ref());
    }
    #[inline]
    fn new_tensor_from_native<T, D>(&self, native: T) -> Tensor<DT, D>
    where
        T: TensorBase<Self::Float, D>,
        D: Dims,
    {
        native.into_owned()
    }

    #[inline]
    fn new_input_adaption_buff<D: DimsMore>(&self, _inner_dims: D) -> () {}

    #[inline]
    fn new_output_adaption_buff<D: DimsMore>(&self, _inner_dims: D) -> () {}

    #[inline]
    fn adapt_input<'a, D: Dims>(&self, _buff: &'a mut (), input: TensorView<'a, DT, D>) -> TensorView<'a, DT, D> {
        input
    }

    #[inline]
    fn adapt_output<'a, D: Dims>(&self, _buff: &'a mut (), output: &'a Tensor<DT, D>) -> &'a Tensor<DT, D> {
        output
    }

    #[inline]
    fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }
}

impl<DT: DTypeOps> MatrixMultiplication for CpuBackend<DT> {
    #[inline]
    fn matmul(&self, alpha: DT, a: TensorView2<DT>, ta: bool, b: TensorView2<DT>, tb: bool, beta: DT, c: &mut Tensor2<DT>) {
        DT::matrix_multiply(alpha, &a, ta, &b, tb, beta, c);
    }
}

impl<DT: DTypeOps> BackendOther for CpuBackend<DT> {
    fn column_sum(&self, alpha: DT, a: &Tensor2<DT>, beta: DT, b: &mut Tensor1<DT>) {
        let &Dim2(rows, cols) = a.dims();
        assert_eq!(b.len(), cols);
        let pa = a.as_ref().as_ptr();
        let pb = b.as_mut().as_mut_ptr();
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
    fn add_assign<D>(&self, alpha: DT, a: &Tensor<DT, D>, beta: DT, b: &mut Tensor<DT, D>)
    where
        D: Dims,
    {
        assert_eq!(a.dims(), b.dims());
        for (&ai, bi) in zip(a, b) {
            *bi = alpha * ai + beta * *bi;
        }
    }

    fn sigmoid(&self, activation: &Tensor2<DT>, output: &mut Tensor2<DT>) {
        assert_eq!(activation.dims(), output.dims());
        for (o, &a) in zip(output, activation) {
            *o = DT::ONE / (DT::ONE + (-a).exp());
        }
    }

    fn sigmoid_error(&self, output: &Tensor2<DT>, out_error: &Tensor2<DT>, result: &mut Tensor2<DT>) {
        assert_eq!(output.dims(), result.dims());
        assert_eq!(output.dims(), out_error.dims());
        for ((r, &out), &err) in zip(zip(result, output), out_error) {
            *r = err * (out * (DT::ONE - out))
        }
    }

    fn relu(&self, leak: DT, activation: &Tensor2<DT>, output: &mut Tensor2<DT>) {
        assert_eq!(activation.dims(), output.dims());
        for (o, &a) in zip(output, activation) {
            *o = if a < DT::ZERO { a * leak } else { a }
        }
    }

    fn relu_error(&self, leak: DT, activation: &Tensor2<DT>, out_error: &Tensor2<DT>, result: &mut Tensor2<DT>) {
        assert_eq!(activation.dims(), result.dims());
        assert_eq!(activation.dims(), out_error.dims());
        for ((r, &act), &err) in zip(zip(result, activation), out_error) {
            *r = if act < DT::ZERO { leak * err } else { err };
        }
    }

    fn softmax(&self, activation: &Tensor2<DT>, output: &mut Tensor2<DT>) {
        assert_eq!(activation.dims(), output.dims());
        for (mut output_row, activation_row) in zip(output.iter_major_axis_mut(), activation.iter_major_axis()) {
            // shift the values by -max(inputs) to prevent overflow (does not affect derivative)
            let max = *activation_row
                .iter()
                .max_by(|&a, &b| if a > b { Ordering::Greater } else { Ordering::Less })
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

    fn softmax_error(&self, output: &Tensor2<DT>, out_error: &Tensor2<DT>, result: &mut Tensor2<DT>) {
        let size = output.dims().cols();
        assert_eq!(output.dims(), result.dims());
        let mut temp = self.temp_matrix.borrow_mut();
        temp.resize(DT::ZERO, Dim2(size, size));
        for (mut result_row, (output_row, out_err_row)) in zip(
            result.iter_major_axis_mut(),
            zip(output.iter_major_axis(), out_error.iter_major_axis()),
        ) {
            compute_jacobian_matrix(output_row.as_ref(), temp.deref_mut());
            DT::matrix_multiply(
                DT::ONE,
                &out_err_row.as_row_matrix(),
                false,
                temp.deref(),
                false,
                DT::ZERO,
                &mut result_row.as_row_matrix_mut(),
            );
        }
    }

    fn mean_squared_error(
        &self,
        output: &Tensor2<DT>,
        expected: TensorView2<DT>,
        result: &mut Tensor1<DT>,
        result_deriv: &mut Tensor2<DT>,
    ) {
        debug_assert_eq!(output.dims().rows(), result.len());
        debug_assert_eq!(output.dims(), expected.dims());
        debug_assert_eq!(output.dims(), result_deriv.dims());
        for (r, (rd_row, (o_row, e_row))) in zip(
            result,
            zip(
                result_deriv.iter_major_axis_mut(),
                zip(output.iter_major_axis(), expected.iter_major_axis()),
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

    #[inline]
    fn flush(&self) {}
    #[inline]
    fn sync(&self) {}

    fn accum_confusion_matrix_multiclass(&self, matrix: &mut Tensor2<DT>, output: &Tensor2<DT>, expected: TensorView2<DT>) {
        for (output_row, expected_row) in zip(output.iter_major_axis(), expected.iter_major_axis()) {
            let out_idx = argmax(output_row.as_ref());
            let expected_idx = argmax(expected_row.as_ref());
            matrix[[expected_idx, out_idx]] += DT::ONE;
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
