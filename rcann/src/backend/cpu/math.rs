use crate::dtype::DTypeFloat;
use crate::tensor::{Dim2, ITensor, Tensor2, TensorBase, TensorBaseMut};
use std::cmp::Ordering;

pub fn compute_jacobian_matrix<T: DTypeFloat>(a: &[T], b: &mut Tensor2<T>) {
    let size = a.len();
    assert_eq!(b.dims(), &Dim2(size, size));
    let pa = a.as_ptr();
    let pb = b.as_mut().as_mut_ptr();
    let sizei = size as isize;
    unsafe {
        for i in 0..sizei {
            let pbi = pb.offset(sizei * i);
            let ai = *pa.offset(i);
            for j in 0..sizei {
                if i == j {
                    *pbi.offset(j) = ai * (T::ONE - ai);
                } else {
                    *pbi.offset(j) = -ai * *pa.offset(j);
                }
            }
        }
    }
}

pub fn argmax<T: Copy + PartialOrd>(a: &[T]) -> usize {
    a.iter()
        .enumerate()
        .max_by(
            |&(_, &a), &(_, &b)| {
                if a < b {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            },
        )
        .expect("expected at least one element")
        .0
}

pub trait DTypeOps: DTypeFloat {
    fn matrix_multiply<A, B, C>(alpha: Self, a: &A, ta: bool, b: &B, tb: bool, beta: Self, c: &mut C)
    where
        A: TensorBase<Self, Dim2>,
        B: TensorBase<Self, Dim2>,
        C: TensorBaseMut<Self, Dim2>;
}

macro_rules! implement_dtype_ops {
    ($t: ident, $g: ident) => {
        impl DTypeOps for $t {
            fn matrix_multiply<A, B, C>(alpha: Self, a: &A, ta: bool, b: &B, tb: bool, beta: Self, c: &mut C)
            where
                A: TensorBase<Self, Dim2>,
                B: TensorBase<Self, Dim2>,
                C: TensorBaseMut<Self, Dim2>,
            {
                let &Dim2(a_rows, a_cols) = a.dims();
                let &Dim2(b_rows, b_cols) = b.dims();
                let &Dim2(_, c_cols) = c.dims();
                let (m, k, rsa, csa) = if ta {
                    (a_cols, a_rows, 1, a_cols as isize)
                } else {
                    (a_rows, a_cols, a_cols as isize, 1)
                };
                let (n, rsb, csb) = if tb {
                    assert_eq!(b_cols, k);
                    (b_rows, 1, b_cols as isize)
                } else {
                    assert_eq!(b_rows, k);
                    (b_cols, b_cols as isize, 1)
                };
                assert_eq!(c.dims(), &Dim2(m, n));
                let (rsc, csc) = (c_cols as isize, 1);
                unsafe {
                    matrixmultiply::$g(
                        m,
                        k,
                        n,
                        alpha,
                        a.as_ref().as_ptr(),
                        rsa,
                        csa,
                        b.as_ref().as_ptr(),
                        rsb,
                        csb,
                        beta,
                        c.as_mut().as_mut_ptr(),
                        rsc,
                        csc,
                    );
                }
            }
        }
    };
}

implement_dtype_ops!(f32, sgemm);
implement_dtype_ops!(f64, dgemm);

#[cfg(feature = "half")]
impl DTypeOps for half::f16 {
    fn matrix_multiply<A, B, C>(alpha: Self, a: &A, ta: bool, b: &B, tb: bool, beta: Self, mut c: &mut C)
    where
        A: TensorBase<Self, Dim2>,
        B: TensorBase<Self, Dim2>,
        C: TensorBaseMut<Self, Dim2>
    {
        let a32 = half_to_f32_tensor(a);
        let b32 = half_to_f32_tensor(b);
        let mut c32 = half_to_f32_tensor(c);
        f32::matrix_multiply(
            alpha.to_f32(),
            &a32,
            ta,
            &b32,
            tb,
            beta.to_f32(),
            &mut c32
        );
        copy_f32_slice_to_half_slice(c32.view().as_ref(), c.view_mut().as_mut());
    }
}

#[cfg(feature = "half")]
fn half_to_f32_tensor<S: TensorBase<half::f16, Dim2>>(src: &S) -> Tensor2<f32> {
    let mut vec = Vec::<f32>::with_capacity(src.len());
    unsafe {
        let mut src_ptr = src.view().as_ref().as_ptr();
        let mut dest_ptr = vec.as_mut_ptr();
        for _ in 0..src.len() {
            *dest_ptr = half::f16::to_f32(*src_ptr);
            src_ptr = src_ptr.add(1);
            dest_ptr = dest_ptr.add(1);
        }
        vec.set_len(src.len());
    }
    Tensor2::from_vec(vec, src.dims().clone())
}

#[cfg(feature = "half")]
fn copy_f32_slice_to_half_slice(src: &[f32], dest: &mut [half::f16]) {
    assert_eq!(src.len(), dest.len());
    unsafe {
        let mut src_ptr = src.as_ptr();
        let mut dest_ptr = dest.as_mut_ptr();
        for _ in 0..src.len() {
            *dest_ptr = half::f16::from_f32(*src_ptr);
            src_ptr = src_ptr.add(1);
            dest_ptr = dest_ptr.add(1);
        }
    }
}

#[cfg(test)]
mod test {
    use crate::backend::cpu::math::DTypeOps;
    use crate::tensor;
    use crate::tensor::{Dim2, Tensor2};

    #[test]
    fn test_mat_mul() {
        let a = tensor![[1., 2., 3.], [4., 5., 6.]];

        let b = tensor![[7., 8.], [9., 10.], [11., 12.]];

        let c = tensor![[0.5, 1.], [1., 0.25]];

        let mut r2x2 = Tensor2::zeroed(Dim2(2, 2));
        let mut r2x3 = Tensor2::zeroed(Dim2(2, 3));
        let mut r3x2 = Tensor2::zeroed(Dim2(3, 2));
        let mut r3x3 = Tensor2::zeroed(Dim2(3, 3));

        // various combinations of A X B

        r2x2.fill(100.); // existing values should be ignored
        f32::matrix_multiply(1.0, &a, false, &b, false, 0.0, &mut r2x2);
        assert_eq!(r2x2, tensor![[58., 64.], [139., 154.]]);

        r2x2.fill(0.);
        f32::matrix_multiply(0.5, &a, false, &b, false, 0.0, &mut r2x2);
        assert_eq!(r2x2, tensor![[29., 32.], [69.5, 77.]]);

        r2x2.fill(1.);
        f32::matrix_multiply(1.0, &a, false, &b, false, 5.0, &mut r2x2);
        assert_eq!(r2x2, tensor![[63., 69.], [144., 159.]]);

        r2x2.fill(1.);
        f32::matrix_multiply(0.5, &a, false, &b, false, 5.0, &mut r2x2);
        assert_eq!(r2x2, tensor![[34., 37.], [74.5, 82.]]);

        // B X A

        r3x3.fill(100.); // existing values should be ignored
        f32::matrix_multiply(1.0, &b, false, &a, false, 0.0, &mut r3x3);
        assert_eq!(r3x3, tensor![[39., 54., 69.], [49., 68., 87.], [59., 82., 105.]]);

        // C X Bt

        r2x3.fill(100.); // existing values should be ignored
        f32::matrix_multiply(1.0, &c, false, &b, true, 0.0, &mut r2x3);
        assert_eq!(r2x3, tensor![[11.5, 14.5, 17.5], [9., 11.5, 14.]]);

        // At X C

        r3x2.fill(100.); // existing values should be ignored
        f32::matrix_multiply(1.0, &a, true, &c, false, 0.0, &mut r3x2);
        assert_eq!(r3x2, tensor![[4.5, 2.], [6., 3.25], [7.5, 4.5]]);
    }

    #[test]
    #[cfg(feature = "half")]
    fn test_mat_mul_f16() {
        use half::f16;
        use crate::tensor_f16;

        let a = tensor_f16![[1., 2., 3.], [4., 5., 6.]];

        let b = tensor_f16![[7., 8.], [9., 10.], [11., 12.]];

        let c = tensor_f16![[0.5, 1.], [1., 0.25]];

        let mut r2x2 = Tensor2::<f16>::zeroed(Dim2(2, 2));
        let mut r2x3 = Tensor2::<f16>::zeroed(Dim2(2, 3));
        let mut r3x2 = Tensor2::<f16>::zeroed(Dim2(3, 2));
        let mut r3x3 = Tensor2::<f16>::zeroed(Dim2(3, 3));

        // various combinations of A X B

        r2x2.fill(f16::from_f32(100.)); // existing values should be ignored
        f16::matrix_multiply(f16::from_f32(1.0), &a, false, &b, false, f16::from_f32(0.0), &mut r2x2);
        assert_eq!(r2x2, tensor_f16![[58., 64.], [139., 154.]]);

        r2x2.fill(f16::from_f32(0.));
        f16::matrix_multiply(f16::from_f32(0.5), &a, false, &b, false, f16::from_f32(0.0), &mut r2x2);
        assert_eq!(r2x2, tensor_f16![[29., 32.], [69.5, 77.]]);

        r2x2.fill(f16::from_f32(1.));
        f16::matrix_multiply(f16::from_f32(1.0), &a, false, &b, false, f16::from_f32(5.0), &mut r2x2);
        assert_eq!(r2x2, tensor_f16![[63., 69.], [144., 159.]]);

        r2x2.fill(f16::from_f32(1.));
        f16::matrix_multiply(f16::from_f32(0.5), &a, false, &b, false, f16::from_f32(5.0), &mut r2x2);
        assert_eq!(r2x2, tensor_f16![[34., 37.], [74.5, 82.]]);

        // B X A

        r3x3.fill(f16::from_f32(100.)); // existing values should be ignored
        f16::matrix_multiply(f16::from_f32(1.0), &b, false, &a, false, f16::from_f32(0.0), &mut r3x3);
        assert_eq!(r3x3, tensor_f16![[39., 54., 69.], [49., 68., 87.], [59., 82., 105.]]);

        // C X Bt

        r2x3.fill(f16::from_f32(100.)); // existing values should be ignored
        f16::matrix_multiply(f16::from_f32(1.0), &c, false, &b, true, f16::from_f32(0.0), &mut r2x3);
        assert_eq!(r2x3, tensor_f16![[11.5, 14.5, 17.5], [9., 11.5, 14.]]);

        // At X C

        r3x2.fill(f16::from_f32(100.)); // existing values should be ignored
        f16::matrix_multiply(f16::from_f32(1.0), &a, true, &c, false, f16::from_f32(0.0), &mut r3x2);
        assert_eq!(r3x2, tensor_f16![[4.5, 2.], [6., 3.25], [7.5, 4.5]]);
    }

}
