use crate::tensor::{Dims, ITensorBase, Tensor, TensorBase, TensorBaseMut};
use crate::dtype::DType;

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
    fn matrix_multiply<A, B, C>(alpha: Self, a: &A, ta: bool, b: &B, tb: bool, beta: Self, c: &mut C, tc: bool) where A: TensorBase<Self>, B: TensorBase<Self>, C: TensorBaseMut<Self>;
}

impl DTypeOps for f32 {
    fn matrix_multiply<A, B, C>(alpha: Self, a: &A, ta: bool, b: &B, tb: bool, beta: Self, c: &mut C, tc: bool) where A: TensorBase<Self>, B: TensorBase<Self>, C: TensorBaseMut<Self> {
        let (a_rows, a_cols) = a.dims().unwrap_2d();
        let (b_rows, b_cols) = b.dims().unwrap_2d();
        let (c_rows, c_cols) = c.dims().unwrap_2d();
        let (m, k, rsa, csa) = if ta {
            (a_cols, a_rows, 1, a_rows as isize)
        } else {
            (a_rows, a_cols, a_cols as isize, 1)
        };
        let (n, rsb, csb) = if tb {
            assert_eq!(b_cols, k);
            (b_rows, 1, b_rows as isize)
        } else {
            assert_eq!(b_rows, k);
            (b_cols, b_cols as isize, 1)
        };
        let (rsc, csc) = if tc {
            assert_eq!(c.dims(), &Dims::D2(n, m));
            (1, c_rows as isize)
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
}
/*
macro_rules! implement_dtype_ops {
    ($t: ident, $g: ident) => {
        impl DTypeOps for $t {
            fn matmul(alpha: $t, a: &TensorView<$t>, ta: bool, b: &TensorView<$t>, tb: bool, beta: $t, c: &mut TensorViewMut<$t>, tc: bool) {
                let (m, k, rsa, csa) = if ta {
                    (a.cols(), a.rows(), 1, a.rows() as isize)
                } else {
                    (a.rows(), a.cols(), a.cols() as isize, 1)
                };
                let (n, rsb, csb) = if tb {
                    assert_eq!(b.cols(), k);
                    (b.rows(), 1, b.rows() as isize)
                } else {
                    assert_eq!(b.rows(), k);
                    (b.cols(), b.cols() as isize, 1)
                };
                let (rsc, csc) = if tc {
                    assert_eq!(c.dim(), (n, m));
                    (1, c.rows() as isize)
                } else {
                    assert_eq!(c.dim(), (m, n));
                    (c.cols() as isize, 1)
                };
                unsafe {
                    matrixmultiply::$g(
                        m, k, n,
                        alpha,
                        a.as_ptr(), rsa, csa,
                        b.as_ptr(), rsb, csb,
                        beta,
                        c.as_mut_ptr(), rsc, csc
                    );
                }
            }
        }
    };
}

implement_dtype_ops!(f32, sgemm);
implement_dtype_ops!(f64, dgemm);
*/