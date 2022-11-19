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
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Dim3(pub usize, pub usize, pub usize);

pub trait Dims: Copy + Debug + Eq + Display {
    const N: usize;
    type Less: Dims;
    fn first(&self) -> usize;
    fn tensor_len(&self) -> usize;
    fn as_vec(&self) -> Vec<usize>;
    fn without_first_axis(&self) -> Self::Less;
    fn with_resized_first_axis(&self, size: usize) -> Self;
}

impl Display for Dim0 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("()")
    }
}

impl Dims for Dim0 {
    const N: usize = 0;
    type Less = Self;
    #[inline]
    fn first(&self) -> usize {
        1
    }
    #[inline]
    fn tensor_len(&self) -> usize {
        1
    }
    fn as_vec(&self) -> Vec<usize> {
        Vec::new()
    }
    fn without_first_axis(&self) -> Self::Less {
        Dim0
    }
    fn with_resized_first_axis(&self, _size: usize) -> Self {
        Dim0
    }
}

impl Dims for Dim1 {
    const N: usize = 1;
    type Less = Dim0;
    #[inline]
    fn first(&self) -> usize {
        self.0
    }
    #[inline]
    fn tensor_len(&self) -> usize {
        self.0
    }
    fn as_vec(&self) -> Vec<usize> {
        vec![self.0]
    }
    fn without_first_axis(&self) -> Self::Less {
        Dim0
    }
    fn with_resized_first_axis(&self, size: usize) -> Self {
        Dim1(size)
    }
}

impl Display for Dim1 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_char('(')?;
        Display::fmt(&self.0, f)?;
        f.write_char(')')
    }
}

impl Dims for Dim2 {
    const N: usize = 2;
    type Less = Dim1;
    #[inline]
    fn first(&self) -> usize {
        self.0
    }
    #[inline]
    fn tensor_len(&self) -> usize {
        self.0 * self.1
    }
    fn as_vec(&self) -> Vec<usize> {
        vec![self.0, self.1]
    }

    fn without_first_axis(&self) -> Self::Less {
        Dim1(self.1)
    }

    fn with_resized_first_axis(&self, size: usize) -> Self {
        Dim2(size, self.1)
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

impl Dims for Dim3 {
    const N: usize = 3;
    type Less = Dim2;
    #[inline]
    fn first(&self) -> usize {
        self.0
    }
    #[inline]
    fn tensor_len(&self) -> usize {
        self.0 * self.1 * self.2
    }

    fn as_vec(&self) -> Vec<usize> {
        vec![self.0, self.1, self.2]
    }

    fn without_first_axis(&self) -> Self::Less {
        Dim2(self.1, self.2)
    }

    fn with_resized_first_axis(&self, size: usize) -> Self {
        Dim3(size, self.1, self.2)
    }
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
