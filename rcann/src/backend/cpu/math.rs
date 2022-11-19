use crate::dtype::DType;
use crate::tensor::{Dims, ITensor, Tensor, TensorBase, TensorBaseMut};

pub fn compute_jacobian_matrix<T: DType>(a: &[T], b: &mut Tensor<T>) {
    let size = a.len();
    assert_eq!(b.dims(), &Dims::D2(size, size));
    let pa = a.as_ptr();
    let pb = b.as_mut_ptr();
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

pub trait DTypeOps: DType {
    fn matrix_multiply<A, B, C>(
        alpha: Self,
        a: &A,
        ta: bool,
        b: &B,
        tb: bool,
        beta: Self,
        c: &mut C,
        tc: bool,
    ) where
        A: TensorBase<Self>,
        B: TensorBase<Self>,
        C: TensorBaseMut<Self>;
}

/*
impl DTypeOps for f32 {
    fn matrix_multiply<A, B, C>(alpha: Self, a: &A, ta: bool, b: &B, tb: bool, beta: Self, c: &mut C, tc: bool) where A: TensorBase<Self>, B: TensorBase<Self>, C: TensorBaseMut<Self> {
        let (a_rows, a_cols) = a.dims().unwrap_2d();
        let (b_rows, b_cols) = b.dims().unwrap_2d();
        let (_, c_cols) = c.dims().unwrap_2d();
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
        let (rsc, csc) = if tc {
            assert_eq!(c.dims(), &Dims::D2(n, m));
            (1, c_cols as isize)
        } else {
            assert_eq!(c.dims(), &Dims::D2(m, n));
            (c_cols as isize, 1)
        };
        unsafe {
            matrixmultiply::sgemm(
                m, k, n,
                alpha,
                a.as_ptr(), rsa, csa,
                b.as_ptr(), rsb, csb,
                beta,
                c.as_mut_ptr(), rsc, csc
            );
        }
    }
}*/

macro_rules! implement_dtype_ops {
    ($t: ident, $g: ident) => {
        impl DTypeOps for $t {
            fn matrix_multiply<A, B, C>(
                alpha: Self,
                a: &A,
                ta: bool,
                b: &B,
                tb: bool,
                beta: Self,
                c: &mut C,
                tc: bool,
            ) where
                A: TensorBase<Self>,
                B: TensorBase<Self>,
                C: TensorBaseMut<Self>,
            {
                let (a_rows, a_cols) = a.dims().unwrap_2d();
                let (b_rows, b_cols) = b.dims().unwrap_2d();
                let (_, c_cols) = c.dims().unwrap_2d();
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
                let (rsc, csc) = if tc {
                    assert_eq!(c.dims(), &Dims::D2(n, m));
                    (1, c_cols as isize)
                } else {
                    assert_eq!(c.dims(), &Dims::D2(m, n));
                    (c_cols as isize, 1)
                };
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

#[cfg(test)]
mod test {
    use crate::backend::cpu::math::DTypeOps;
    use crate::tensor::Tensor;

    macro_rules! assert_slice_equal {
        ($a:ident, $b:expr) => {{
            let b = $b;
            if $a.len() != b.len()
                || !std::iter::zip(&$a, &b).all(|(&i, &j)| (i - j).abs() <= f32::EPSILON)
            {
                let mismatch: Vec<usize> = std::iter::zip(&$a, &b)
                    .enumerate()
                    .filter(|(_, (&i, &j))| (i - j).abs() > f32::EPSILON)
                    .map(|(idx, _)| idx)
                    .collect();
                panic!(
                    "slices not equal: left={:?}, right={:?}, mismatched indexes={:?}",
                    &$a, &b, &mismatch
                );
            }
        }};
    }

    #[test]
    fn test_mat_mul() {
        let a = Tensor::from_vec(vec![1., 2., 3., 4., 5., 6.], (2, 3));

        let b = Tensor::from_vec(vec![7., 8., 9., 10., 11., 12.], (3, 2));

        let c = Tensor::from_vec(vec![0.5, 1., 1., 0.25], (2, 2));

        let mut r2x2 = Tensor::filled(0., (2, 2));
        let mut r2x3 = Tensor::filled(0., (2, 3));
        let mut r3x2 = Tensor::filled(0., (3, 2));
        let mut r3x3 = Tensor::filled(0., (3, 3));

        // various combinations of A X B

        r2x2.fill(100.); // existing values should be ignored
        f32::matrix_multiply(1.0, &a, false, &b, false, 0.0, &mut r2x2, false);
        assert_slice_equal!(r2x2, [58., 64., 139., 154.]);

        r2x2.fill(0.);
        f32::matrix_multiply(0.5, &a, false, &b, false, 0.0, &mut r2x2, false);
        assert_slice_equal!(r2x2, [29., 32., 69.5, 77.]);

        r2x2.fill(1.);
        f32::matrix_multiply(1.0, &a, false, &b, false, 5.0, &mut r2x2, false);
        assert_slice_equal!(r2x2, [63., 69., 144., 159.]);

        r2x2.fill(1.);
        f32::matrix_multiply(0.5, &a, false, &b, false, 5.0, &mut r2x2, false);
        assert_slice_equal!(r2x2, [34., 37., 74.5, 82.]);

        // B X A

        r3x3.fill(100.); // existing values should be ignored
        f32::matrix_multiply(1.0, &b, false, &a, false, 0.0, &mut r3x3, false);
        assert_slice_equal!(r3x3, [39., 54., 69., 49., 68., 87., 59., 82., 105.]);

        // C X Bt

        r2x3.fill(100.); // existing values should be ignored
        f32::matrix_multiply(1.0, &c, false, &b, true, 0.0, &mut r2x3, false);
        assert_slice_equal!(r2x3, [11.5, 14.5, 17.5, 9., 11.5, 14.]);

        // At X C

        r3x2.fill(100.); // existing values should be ignored
        f32::matrix_multiply(1.0, &a, true, &c, false, 0.0, &mut r3x2, false);
        assert_slice_equal!(r3x2, [4.5, 2., 6., 3.25, 7.5, 4.5]);

        // At X C -> Rt

        r2x3.fill(100.); // existing values should be ignored
        f32::matrix_multiply(1.0, &a, true, &c, false, 0.0, &mut r2x3, true);
        assert_slice_equal!(r2x3, [4.5, 6., 7.5, 2., 3.25, 4.5]);
    }
}
