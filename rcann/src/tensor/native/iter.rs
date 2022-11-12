use crate::dtype::DType;
use crate::tensor::{Dims, TensorView, TensorViewMut};

pub struct TensorIter<'a, T: DType> {
    rest: &'a [T],
    stride: usize,
    out_dims: Dims,
}

impl<'a, T: DType> TensorIter<'a, T> {
    pub(crate) unsafe fn new_unchecked(data: &'a [T], out_dims: Dims) -> Self {
        debug_assert!(data.len() % out_dims.tensor_len() == 0);
        let stride = out_dims.tensor_len();
        TensorIter {
            rest: data,
            stride,
            out_dims,
        }
    }
}

impl<'a, T: DType> Iterator for TensorIter<'a, T> {
    type Item = TensorView<'a, T>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.rest.is_empty() {
            None
        } else {
            unsafe {
                let (chunk, rest) = self.rest.split_at_unchecked(self.stride);
                self.rest = rest;
                Some(TensorView::from_slice_unchecked(chunk, self.out_dims.clone()))
            }
        }
    }
}

pub struct TensorIterMut<'a, T: 'a + DType> {
    rest: &'a mut [T],
    stride: usize,
    out_dims: Dims,
}

impl<'a, T: DType> TensorIterMut<'a, T> {
    pub(crate) unsafe fn new_unchecked(data: &'a mut [T], out_dims: Dims) -> Self {
        debug_assert!(data.len() % out_dims.tensor_len() == 0);
        let stride = out_dims.tensor_len();
        TensorIterMut {
            rest: data,
            stride,
            out_dims,
        }
    }
}

impl<'a, T: 'a + DType> Iterator for TensorIterMut<'a, T> {
    type Item = TensorViewMut<'a, T>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.rest.is_empty() {
            None
        } else {
            unsafe {
                // For some reason, split_at_mut results in lifetime errors?
                // let (chunk, rest) = self.rest.split_at_mut_unchecked(self.stride);
                let ptr = self.rest.as_mut_ptr();
                let chunk = std::slice::from_raw_parts_mut(ptr, self.stride);
                let rest_len = self.rest.len() - self.stride;
                self.rest = std::slice::from_raw_parts_mut(ptr.add(self.stride), rest_len);
                Some(TensorViewMut::from_slice_unchecked(chunk, self.out_dims.clone()))
            }
        }
    }
}

/*
pub struct TensorChunkIter<'a, T: DType> {
    iter: TensorIter<'a, T>,
    remainder: Option<TensorView<'a, T>>,
}

impl<'a, T: DType> TensorChunkIter<'a, T> {
    pub(crate) unsafe fn new_unchecked(data: &'a [T], chunk_dims: Dims) -> Self {
        debug_assert!(data.len() % chunk_dims.first_axis_stride() == 0);
        let remainder_len = data.len() % chunk_dims.tensor_len();
        if remainder_len > 0 {
            let (data, remainder) = data.split_at_unchecked(data.len() - remainder_len);
            let rem_first_axis_len = remainder_len / chunk_dims.first_axis_stride();
            let rem_dim = chunk_dims.with_resized_first_axis(rem_first_axis_len);
            TensorChunkIter {
                iter: TensorIter::new_unchecked(data, chunk_dims),
                remainder: Some(TensorView::from_slice_unchecked(remainder, rem_dim))
            }
        } else {
            TensorChunkIter {
                iter: TensorIter::new_unchecked(data, chunk_dims),
                remainder: None
            }
        }
    }
}

impl<'a, T: DType> Iterator for TensorChunkIter<'a, T> {
    type Item = TensorView<'a, T>;
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(next) = self.iter.next() {
            Some(next)
        } else {
            self.remainder.take()
        }
    }
}

pub struct TensorChunkIterMut<'a, T: DType> {
    iter: TensorIterMut<'a, T>,
    remainder: Option<TensorViewMut<'a, T>>,
}

impl<'a, T: DType> TensorChunkIterMut<'a, T> {
    pub(crate) unsafe fn new_unchecked(data: &'a mut [T], chunk_dims: Dims) -> Self {
        debug_assert!(data.len() % chunk_dims.first_axis_stride() == 0);
        let remainder_len = data.len() % chunk_dims.tensor_len();
        if remainder_len > 0 {
            let (data, remainder) = data.split_at_mut_unchecked(data.len() - remainder_len);
            let rem_first_axis_len = remainder_len / chunk_dims.first_axis_stride();
            let rem_dim = chunk_dims.with_resized_first_axis(rem_first_axis_len);
            TensorChunkIterMut {
                iter: TensorIterMut::new_unchecked(data, chunk_dims),
                remainder: Some(TensorViewMut::from_slice_unchecked(remainder, rem_dim))
            }
        } else {
            TensorChunkIterMut {
                iter: TensorIterMut::new_unchecked(data, chunk_dims),
                remainder: None
            }
        }
    }
}

impl<'a, T: DType> Iterator for TensorChunkIterMut<'a, T> {
    type Item = TensorViewMut<'a, T>;
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(next) = self.iter.next() {
            Some(next)
        } else {
            self.remainder.take()
        }
    }
}
*/