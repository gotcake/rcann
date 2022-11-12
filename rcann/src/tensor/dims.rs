use std::fmt::{Display, Formatter};
use std::ops::Index;

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Dims {
    D0,
    D1(usize),
    D2(usize, usize),
    D3(usize, usize, usize),
}

impl Dims {
    pub fn len(&self) -> usize {
        use Dims::*;
        match self {
            D0 => 0,
            D1(_) => 1,
            D2(_, _) => 2,
            D3(_, _, _) => 3,
        }
    }

    pub fn tensor_len(&self) -> usize {
        use Dims::*;
        match self {
            D0 => 1,
            &D1(a) => a,
            &D2(a, b) => a * b,
            &D3(a, b, c) => a * b * c,
        }
    }

    pub fn first(&self) -> usize {
        use Dims::*;
        match self {
            D0 => 0,
            &D1(a) => a,
            &D2(a, _) => a,
            &D3(a, _, _) => a,
        }
    }

    pub fn as_vec(&self) -> Vec<usize> {
        use Dims::*;
        match self {
            D0 => Vec::new(),
            &D1(a) => vec![a],
            &D2(a, b) => vec![a, b],
            &D3(a, b, c) => vec![a, b, c],
        }
    }

    pub fn is_scalar(&self) -> bool {
        use Dims::*;
        match self {
            D0 => true,
            _ => false,
        }
    }

    pub fn is_1d(&self) -> bool {
        use Dims::*;
        match self {
            D1(_) => true,
            _ => false,
        }
    }

    pub fn is_2d(&self) -> bool {
        use Dims::*;
        match self {
            D2(_, _) => true,
            _ => false,
        }
    }

    pub fn is_3d(&self) -> bool {
        use Dims::*;
        match self {
            D3(_, _, _) => true,
            _ => false,
        }
    }

    pub fn as_1d(&self) -> Option<usize> {
        use Dims::*;
        match self {
            &D1(a) => Some(a),
            _ => None,
        }
    }

    #[inline]
    pub fn unwrap_1d(&self) -> usize {
        use Dims::*;
        match self {
            &D1(a) => a,
            _ => panic!("Expected tensor with 1 dimension, but dimensions are: {self}"),
        }
    }

    pub fn as_2d(&self) -> Option<(usize, usize)> {
        use Dims::*;
        match self {
            &D2(a, b) => Some((a, b)),
            _ => None,
        }
    }

    #[inline]
    pub fn unwrap_2d(&self) -> (usize, usize) {
        use Dims::*;
        match self {
            &D2(a, b) => (a, b),
            _ => panic!("Expected tensor with 2 dimensions, but dimensions are: {self}"),
        }
    }

    pub fn as_3d(&self) -> Option<(usize, usize, usize)> {
        use Dims::*;
        match self {
            &D3(a, b, c) => Some((a, b, c)),
            _ => None,
        }
    }

    #[inline]
    pub fn unwrap_3d(&self) -> (usize, usize, usize) {
        match self.as_3d() {
            None => panic!("Expected tensor with 3 dimensions, but dimensions are: {self}"),
            Some(d) => d,
        }
    }

    pub fn with_resized_first_axis(&self, first: usize) -> Dims {
        use Dims::*;
        match self {
            D0 => D0,
            D1(_) => D1(first),
            &D2(_, b) => D2(first, b),
            &D3(_, b, c) => D3(first, b, c),
        }
    }

    pub fn without_first_axis(&self) -> Dims {
        use Dims::*;
        match self {
            D0 => D0,
            D1(_) => D0,
            &D2(_, b) => D1(b),
            &D3(_, b, c) => D2(b, c),
        }
    }

    pub fn first_axis_stride(&self) -> usize {
        use Dims::*;
        match self {
            D0 => 0,
            D1(_) => 1,
            &D2(_, b) => b,
            &D3(_, b, c) => b * c,
        }
    }
}

impl Into<Dims> for usize {
    #[inline]
    fn into(self) -> Dims {
        Dims::D1(self)
    }
}

impl Into<Dims> for (usize, usize) {
    #[inline]
    fn into(self) -> Dims {
        let (a, b) = self;
        Dims::D2(a, b)
    }
}

impl Into<Dims> for (usize, usize, usize) {
    #[inline]
    fn into(self) -> Dims {
        let (a, b, c) = self;
        Dims::D3(a, b, c)
    }
}

impl Into<Dims> for &Dims {
    #[inline]
    fn into(self) -> Dims {
        self.clone()
    }
}

impl Index<usize> for Dims {
    type Output = usize;
    fn index(&self, index: usize) -> &Self::Output {
        use Dims::*;
        match (self, index) {
            (D0, _) => &1,
            (D1(a), 0) => a,
            (D2(a, _), 0) => a,
            (D2(_, b), 1) => b,
            (D3(a, _, _), 0) => a,
            (D3(_, b, _), 1) => b,
            (D3(_, _, c), 2) => c,
            _ => panic!("Invalid index {} for {:?}", index, self),
        }
    }
}

impl Display for Dims {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use Dims::*;
        match self {
            D0 => f.write_str("()"),
            &D1(a) => write!(f, "({a})"),
            &D2(a, b) => write!(f, "({a}, {b})"),
            &D3(a, b, c) => write!(f, "({a}, {b}, {c})"),
        }
    }
}
