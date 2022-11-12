use std::iter::zip;

#[inline]
pub(crate) fn resize_if_needed(vec: &mut Vec<f32>, len: usize) {
    if vec.len() != len {
        if vec.capacity() < len {
            vec.reserve_exact(len - vec.capacity());
        }
        let old_len = vec.len();
        unsafe { vec.set_len(len); }
        if old_len < vec.len() {
            vec[old_len..len].fill(0.0);
        }
    }
}

#[inline]
pub(crate) fn generic_column_sum(r: usize, c: usize, alpha: f32, beta: f32, source: &[f32], target: &mut [f32]) {
    debug_assert_eq!(source.len(), r * c);
    debug_assert_eq!(target.len(), c);
    let s = source.as_ptr();
    let t = target.as_mut_ptr();
    let r = r as isize;
    let c = c as isize;
    for i in 0..c {
        let mut sum = 0.0;
        for j in 0..r {
            sum += unsafe { *s.offset(j * c + i) };
        }
        unsafe {
            let p = t.offset(i);
            *p = sum * alpha + *p * beta;
        }
    }
}

#[inline(always)]
pub(crate) fn sum(arr: &[f32]) -> f32 {
    arr.iter().fold(0.0, |acc, &x| acc + x)
}

#[inline]
pub(crate) fn generic_row_sum(r: usize, c: usize, alpha: f32, beta: f32, source: &[f32], target: &mut [f32]) {
    debug_assert_eq!(source.len(), r * c);
    debug_assert_eq!(target.len(), r);
    for (t, row) in zip( target, source.chunks_exact(c)) {
        *t = alpha * sum(row) + *t * beta;
    }
}

#[inline]
pub(crate) fn sub_assign(source: &[f32], target: &mut [f32]) {
    debug_assert_eq!(source.len(), target.len());
    for (t, &s) in zip(target, source) {
        *t -= s;
    }
}

#[inline]
pub(crate) fn mat_mul(m: usize, k: usize, n: usize, alpha: f32, beta: f32, a: &[f32], b: &[f32], c: &mut [f32]) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);
    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            alpha,
            a.as_ptr(),
            k as isize,
            1,
            b.as_ptr(),
            n as isize,
            1,
            beta,
            c.as_mut_ptr(),
            n as isize,
            1
        );
    }
}

#[inline]
pub(crate) fn mat_mul_b_transpose(m: usize, k: usize, n: usize, alpha: f32, beta: f32, a: &[f32], b: &[f32], c: &mut [f32]) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);
    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            alpha,
            a.as_ptr(),
            k as isize,
            1,
            b.as_ptr(),
            1,
            k as isize,
            beta,
            c.as_mut_ptr(),
            n as isize,
            1
        );
    }
}

#[inline]
pub(crate) fn mat_mul_a_transpose(m: usize, k: usize, n: usize, alpha: f32, beta: f32, a: &[f32], b: &[f32], c: &mut [f32]) {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(b.len(), k * n);
    debug_assert_eq!(c.len(), m * n);
    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            alpha,
            a.as_ptr(),
            1,
            m as isize,
            b.as_ptr(),
            n as isize,
            1,
            beta,
            c.as_mut_ptr(),
            n as isize,
            1
        );
    }
}

#[cfg(test)]
mod test {
    use std::iter::zip;
    use std::usize;
    use crate::raw::util::{mat_mul, mat_mul_a_transpose, mat_mul_b_transpose};

    macro_rules! assert_slice_equal {
        ($a:ident, $b:expr) => {
            {
                let b = $b;
                if $a.len() != b.len() || !std::iter::zip(&$a, &b).all(|(&i, &j)| (i - j).abs() <= f32::EPSILON) {
                    let mismatch: Vec<usize> = std::iter::zip(&$a, &b)
                        .enumerate()
                        .filter(|(idx, (&i, &j))| (i - j).abs() > f32::EPSILON )
                        .map(|(idx, _)| idx)
                        .collect();
                    panic!("slices not equal: left={:?}, right={:?}, mismatched indexes={:?}", &$a, &b, &mismatch);
                }
            }
        }
    }

    #[test]
    fn test_mat_mul() {

        let a = [
            1., 2., 3.,
            4., 5., 6.
        ];
        let b = [
            7., 8.,
            9., 10.,
            11., 12.
        ];
        let c = [
            0.5, 1.,
            1., 0.25
        ];

        let mut r4 = [0.; 4];
        let mut r6 = [0.; 6];
        let mut r9 = [0.; 9];

        // various combinations of A X B

        r4.fill(100.); // existing values should be ignored
        mat_mul(2, 3, 2, 1.0, 0.0, &a, &b, &mut r4);
        assert_slice_equal!(r4, [
            58., 64.,
            139., 154.
        ]);

        r4.fill(0.);
        mat_mul(2, 3, 2, 0.5, 0.0, &a, &b, &mut r4);
        assert_slice_equal!(r4, [
            29., 32.,
            69.5, 77.
        ]);

        r4.fill(1.);
        mat_mul(2, 3, 2, 1.0, 5.0, &a, &b, &mut r4);
        assert_slice_equal!(r4, [
            63., 69.,
            144., 159.
        ]);

        r4.fill(1.);
        mat_mul(2, 3, 2, 0.5, 5.0, &a, &b, &mut r4);
        assert_slice_equal!(r4, [
            34., 37.,
            74.5, 82.
        ]);

        // B X A

        r9.fill(100.); // existing values should be ignored
        mat_mul(3, 2, 3, 1.0, 0.0, &b, &a, &mut r9);
        assert_slice_equal!(r9, [
            39., 54., 69.,
            49., 68., 87.,
            59., 82., 105.
        ]);

        // C X Bt

        r6.fill(100.); // existing values should be ignored
        mat_mul_b_transpose(2, 2, 3, 1.0, 0.0, &c, &b, &mut r6);
        assert_slice_equal!(r6, [
            11.5, 14.5, 17.5,
            9., 11.5, 14.
        ]);

        // At X C

        r6.fill(100.); // existing values should be ignored
        mat_mul_a_transpose(3, 2, 2, 1.0, 0.0, &a, &c, &mut r6);
        assert_slice_equal!(r6, [
            4.5, 2.,
            6., 3.25,
            7.5, 4.5
        ]);


    }

}