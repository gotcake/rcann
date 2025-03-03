use crate::tensor::{Dims, DimsMore, TensorView, TensorViewMut};
use std::marker::PhantomData;
use std::slice::{Chunks, ChunksMut};

// TODO: unsafe for zero sized types
pub struct TensorIter<'a, T: 'a, D: Dims> {
    ptr: *const T,
    end: *const T,
    stride: usize,
    out_dims: D,
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T: 'a, D: Dims> TensorIter<'a, T, D> {
    pub(super) unsafe fn new_unchecked(data: &'a [T], out_dims: D) -> Self {
        debug_assert!((data.len() == 0 && out_dims.tensor_len() == 0) || data.len() % out_dims.tensor_len() == 0);
        let stride = out_dims.tensor_len();
        let range = data.as_ptr_range();
        debug_assert!(range.start <= range.end);
        TensorIter {
            ptr: range.start,
            end: range.end,
            stride,
            out_dims,
            _marker: PhantomData,
        }
    }
}

// TODO: unsafe for zero sized types
pub struct TensorIterMut<'a, T: 'a, D: Dims> {
    ptr: *mut T,
    end: *mut T,
    stride: usize,
    out_dims: D,
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T: 'a, D: Dims> TensorIterMut<'a, T, D> {
    pub(super) unsafe fn new_unchecked(data: &'a mut [T], out_dims: D) -> Self {
        debug_assert!((data.len() == 0 && out_dims.tensor_len() == 0) || data.len() % out_dims.tensor_len() == 0);
        let stride = out_dims.tensor_len();
        let range = data.as_mut_ptr_range();
        debug_assert!(range.start <= range.end);
        TensorIterMut {
            ptr: range.start,
            end: range.end,
            stride,
            out_dims,
            _marker: PhantomData,
        }
    }
}

macro_rules! impl_tensor_iter {
    ($type_name: ident, $item: ident, $slice_from_raw_parts: ident) => {
        impl<'a, T: 'a, D: Dims> $type_name<'a, T, D> {
            #[inline]
            unsafe fn next_unchecked(&mut self) -> $item<'a, T, D> {
                let chunk = std::slice::$slice_from_raw_parts(self.ptr, self.stride);
                self.ptr = self.ptr.add(self.stride);
                debug_assert!(self.ptr <= self.end);
                $item::from_slice_unchecked(chunk, self.out_dims)
            }
        }

        impl<'a, T: 'a, D: Dims> Iterator for $type_name<'a, T, D> {
            type Item = $item<'a, T, D>;
            fn next(&mut self) -> Option<Self::Item> {
                debug_assert!(self.ptr <= self.end);
                if self.ptr == self.end {
                    None
                } else {
                    Some(unsafe { self.next_unchecked() })
                }
            }
            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                let size = self.len();
                (size, Some(size))
            }
            #[inline]
            fn count(self) -> usize
            where
                Self: Sized,
            {
                self.len()
            }
            fn nth(&mut self, n: usize) -> Option<Self::Item> {
                debug_assert!(self.ptr <= self.end);
                unsafe {
                    let new_ptr = self.ptr.add(n * self.stride);
                    if new_ptr >= self.end {
                        self.ptr = self.end;
                        None
                    } else {
                        self.ptr = new_ptr;
                        Some(self.next_unchecked())
                    }
                }
            }
        }

        impl<'a, T: 'a, D: Dims> ExactSizeIterator for $type_name<'a, T, D> {
            #[inline]
            fn len(&self) -> usize {
                if self.ptr == self.end {
                    0
                } else {
                    debug_assert!(self.ptr <= self.end);
                    debug_assert!(self.stride > 0);
                    unsafe { self.end.offset_from(self.ptr) as usize / self.stride }
                }
            }
        }
    };
}

impl_tensor_iter!(TensorIter, TensorView, from_raw_parts);
impl_tensor_iter!(TensorIterMut, TensorViewMut, from_raw_parts_mut);

pub struct TensorChunkIter<'a, T: 'a, D: DimsMore> {
    chunk_iter: Chunks<'a, T>,
    inner_dim: D,
    inner_len: usize,
}

impl<'a, T: 'a, D: DimsMore> TensorChunkIter<'a, T, D> {
    #[inline]
    pub(super) unsafe fn new_unchecked(slice: &'a [T], inner_dim: D, size: usize) -> Self {
        let inner_len = inner_dim.tensor_len();
        Self {
            chunk_iter: slice.chunks(inner_len * size),
            inner_dim,
            inner_len,
        }
    }
}

impl<'a, T: 'a, D: DimsMore> Iterator for TensorChunkIter<'a, T, D> {
    type Item = TensorView<'a, T, D::More>;
    fn next(&mut self) -> Option<Self::Item> {
        self.chunk_iter.next().map(|chunk| {
            debug_assert_eq!(chunk.len() % self.inner_len, 0);
            let dims = self.inner_dim.insert_major(chunk.len() / self.inner_len);
            unsafe { TensorView::from_slice_unchecked(chunk, dims) }
        })
    }
}

pub struct TensorChunkIterMut<'a, T: 'a, D: DimsMore> {
    chunk_iter: ChunksMut<'a, T>,
    inner_dim: D,
    inner_len: usize,
}

impl<'a, T: 'a, D: DimsMore> TensorChunkIterMut<'a, T, D> {
    #[inline]
    pub(super) unsafe fn new_unchecked(slice: &'a mut [T], inner_dim: D, size: usize) -> Self {
        let inner_len = inner_dim.tensor_len();
        Self {
            chunk_iter: slice.chunks_mut(inner_len * size),
            inner_dim,
            inner_len,
        }
    }
}

impl<'a, T: 'a, D: DimsMore> Iterator for TensorChunkIterMut<'a, T, D> {
    type Item = TensorViewMut<'a, T, D::More>;
    fn next(&mut self) -> Option<Self::Item> {
        self.chunk_iter.next().map(|chunk| {
            debug_assert_eq!(chunk.len() % self.inner_len, 0);
            let dims = self.inner_dim.insert_major(chunk.len() / self.inner_len);
            unsafe { TensorViewMut::from_slice_unchecked(chunk, dims) }
        })
    }
}

#[cfg(test)]
mod test {
    use crate::tensor;

    #[test]
    fn test_tensor_iter_1d() {
        let t = tensor![1, 2, 3, 4, 5, 6];
        // TODO
    }
}
