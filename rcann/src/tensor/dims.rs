use std::fmt::{Debug, Display, Formatter, Write};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Dim0;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Dim1(pub usize);

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Dim2(pub usize, pub usize);

impl Dim2 {
    #[inline]
    pub fn rows(&self) -> usize {
        self.0
    }
    #[inline]
    pub fn cols(&self) -> usize {
        self.1
    }
    #[inline]
    pub fn transposed(&self) -> Dim2 {
        Dim2(self.1, self.0)
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Dim3(pub usize, pub usize, pub usize);

pub unsafe trait Dims: Copy + Debug + Eq + Display {
    const N: usize;
    type Less: DimsMore;
    type Array;
    fn major(&self) -> usize;
    fn minor(&self) -> usize;
    fn tensor_len(&self) -> usize;
    fn as_vec(&self) -> Vec<usize>;
    fn remove_major(&self) -> Self::Less;
    fn remove_minor(&self) -> Self::Less;
    fn resize_major(&self, size: usize) -> Self;
    fn map_each<F>(&self, f: F) -> Self
    where
        F: FnMut(usize, usize) -> usize;
    fn as_dim3(&self) -> Dim3;
    fn get_compact_offset(&self, index: &Self::Array) -> usize;
}

pub unsafe trait DimsZero: Dims {
    const ZERO: Self;
}

pub unsafe trait DimsMore: Dims {
    type More: Dims;
    fn insert_major(&self, size: usize) -> Self::More;
}

impl Display for Dim0 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("()")
    }
}

unsafe impl Dims for Dim0 {
    const N: usize = 0;
    type Less = Self;
    type Array = [usize; 0];
    #[inline]
    fn major(&self) -> usize {
        1
    }
    #[inline]
    fn minor(&self) -> usize {
        1
    }
    #[inline]
    fn tensor_len(&self) -> usize {
        1
    }
    fn as_vec(&self) -> Vec<usize> {
        Vec::new()
    }
    #[inline]
    fn remove_major(&self) -> Self::Less {
        Dim0
    }
    #[inline]
    fn remove_minor(&self) -> Self::Less {
        Dim0
    }
    #[inline]
    fn resize_major(&self, _size: usize) -> Self {
        Dim0
    }
    #[inline]
    fn map_each<F>(&self, _f: F) -> Self
    where
        F: FnMut(usize, usize) -> usize,
    {
        Dim0
    }
    #[inline]
    fn as_dim3(&self) -> Dim3 {
        Dim3(1, 1, 1)
    }
    #[inline]
    fn get_compact_offset(&self, index: &Self::Array) -> usize {
        0
    }
}

unsafe impl DimsMore for Dim0 {
    type More = Dim1;
    #[inline]
    fn insert_major(&self, size: usize) -> Dim1 {
        Dim1(size)
    }
}

unsafe impl Dims for Dim1 {
    const N: usize = 1;
    type Less = Dim0;
    type Array = [usize; 1];
    #[inline]
    fn major(&self) -> usize {
        self.0
    }
    #[inline]
    fn minor(&self) -> usize {
        self.0
    }
    #[inline]
    fn tensor_len(&self) -> usize {
        self.0
    }
    fn as_vec(&self) -> Vec<usize> {
        vec![self.0]
    }
    #[inline]
    fn remove_major(&self) -> Self::Less {
        Dim0
    }
    #[inline]
    fn remove_minor(&self) -> Self::Less {
        Dim0
    }
    #[inline]
    fn resize_major(&self, size: usize) -> Self {
        Dim1(size)
    }
    fn map_each<F>(&self, mut f: F) -> Self
    where
        F: FnMut(usize, usize) -> usize,
    {
        Dim1(f(self.0, 0))
    }
    #[inline]
    fn as_dim3(&self) -> Dim3 {
        Dim3(1, 1, self.0)
    }
    #[inline]
    fn get_compact_offset(&self, index: &Self::Array) -> usize {
        index[0]
    }
}

unsafe impl DimsZero for Dim1 {
    const ZERO: Self = Dim1(0);
}

unsafe impl DimsMore for Dim1 {
    type More = Dim2;
    #[inline]
    fn insert_major(&self, size: usize) -> Dim2 {
        Dim2(size, self.0)
    }
}

impl Display for Dim1 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_char('(')?;
        Display::fmt(&self.0, f)?;
        f.write_char(')')
    }
}

unsafe impl Dims for Dim2 {
    const N: usize = 2;
    type Less = Dim1;
    type Array = [usize; 2];
    #[inline]
    fn major(&self) -> usize {
        self.0
    }
    #[inline]
    fn minor(&self) -> usize {
        self.1
    }
    #[inline]
    fn tensor_len(&self) -> usize {
        self.0 * self.1
    }
    fn as_vec(&self) -> Vec<usize> {
        vec![self.0, self.1]
    }
    #[inline]
    fn remove_major(&self) -> Self::Less {
        Dim1(self.1)
    }
    #[inline]
    fn remove_minor(&self) -> Self::Less {
        Dim1(self.0)
    }
    #[inline]
    fn resize_major(&self, size: usize) -> Self {
        Dim2(size, self.1)
    }

    fn map_each<F>(&self, mut f: F) -> Self
    where
        F: FnMut(usize, usize) -> usize,
    {
        Dim2(f(self.0, 0), f(self.1, 1))
    }
    #[inline]
    fn as_dim3(&self) -> Dim3 {
        Dim3(1, self.0, self.1)
    }
    #[inline]
    fn get_compact_offset(&self, index: &Self::Array) -> usize {
        index[0] * self.1 + index[1]
    }
}

unsafe impl DimsZero for Dim2 {
    const ZERO: Self = Dim2(0, 0);
}

unsafe impl DimsMore for Dim2 {
    type More = Dim3;
    #[inline]
    fn insert_major(&self, size: usize) -> Dim3 {
        Dim3(size, self.0, self.1)
    }
}

impl Display for Dim2 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_char('(')?;
        Display::fmt(&self.0, f)?;
        f.write_str(", ")?;
        Display::fmt(&self.1, f)?;
        f.write_char(')')
    }
}

unsafe impl Dims for Dim3 {
    const N: usize = 3;
    type Less = Dim2;
    type Array = [usize; 3];
    #[inline]
    fn major(&self) -> usize {
        self.0
    }
    #[inline]
    fn minor(&self) -> usize {
        self.2
    }
    #[inline]
    fn tensor_len(&self) -> usize {
        self.0 * self.1 * self.2
    }
    fn as_vec(&self) -> Vec<usize> {
        vec![self.0, self.1, self.2]
    }
    #[inline]
    fn remove_major(&self) -> Self::Less {
        Dim2(self.1, self.2)
    }
    #[inline]
    fn remove_minor(&self) -> Self::Less {
        Dim2(self.0, self.1)
    }
    #[inline]
    fn resize_major(&self, size: usize) -> Self {
        Dim3(size, self.1, self.2)
    }

    fn map_each<F>(&self, mut f: F) -> Self
    where
        F: FnMut(usize, usize) -> usize,
    {
        Dim3(f(self.0, 0), f(self.1, 1), f(self.2, 2))
    }
    #[inline]
    fn as_dim3(&self) -> Dim3 {
        *self
    }
    #[inline]
    fn get_compact_offset(&self, index: &Self::Array) -> usize {
        index[0] * self.1 * self.2 + index[1] * self.2 + index[2]
    }
}

unsafe impl DimsZero for Dim3 {
    const ZERO: Self = Dim3(0, 0, 0);
}

impl Display for Dim3 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_char('(')?;
        Display::fmt(&self.0, f)?;
        f.write_str(", ")?;
        Display::fmt(&self.1, f)?;
        f.write_str(", ")?;
        Display::fmt(&self.2, f)?;
        f.write_char(')')
    }
}
