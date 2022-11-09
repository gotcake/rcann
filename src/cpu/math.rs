use crate::cpu::matrix::{MatrixRef, MatrixRefMut};
use crate::dtype::DType;
use crate::tensor::Tensor2;

pub fn jacobian<T: DType>(a: &[T], b: &mut MatrixRefMut<T>) {
    let size = a.len();
    assert_eq!(b.dim(), (size, size));
    let pa = a.as_ptr();
    let pb = b.as_mut_ptr();
    let sizei = size as isize;
    unsafe {
        for i in 0..sizei {
            let pbi = pb.offset(sizei * i);
            let ai = *pa.offset(i);
            for j in 0..sizei {
                if i == j {
                    *pbi.offset(j) = ai * (T::one() - ai);
                } else {
                    *pbi.offset(j) = -ai * *pa.offset(j);
                }
            }
        }
    }
}

pub trait DTypeOps: DType {
    fn matmul(alpha: Self, a: &MatrixRef<Self>, ta: bool, b: &MatrixRef<Self>, tb: bool, beta: Self, c: &mut MatrixRefMut<Self>, tc: bool);
}

macro_rules! implement_dtype_ops {
    ($t: ident, $g: ident) => {
        impl DTypeOps for $t {
            fn matmul(alpha: $t, a: &MatrixRef<$t>, ta: bool, b: &MatrixRef<$t>, tb: bool, beta: $t, c: &mut MatrixRefMut<$t>, tc: bool) {
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
