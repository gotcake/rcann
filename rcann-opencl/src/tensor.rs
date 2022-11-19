use crate::error::Error;
use crate::util::{next_multiple, Result};
use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::event::{Event, CL_COMPLETE};
use opencl3::memory::{Buffer, CL_MEM_READ_WRITE};
use opencl3::types::{cl_double, cl_float, CL_BLOCKING};
use rcann::tensor::{Dim2, Dims, ITensor, Tensor, TensorBase, TensorBaseMut, TensorView, TensorViewMut};
use std::cell::RefCell;
use std::cmp::min;
use std::iter::zip;
use std::ops::Deref;
use std::ptr;
use std::rc::Rc;
use crate::util::MATRIX_PADDING_SIZE;

pub trait OclDType: Copy + Default {}
impl OclDType for cl_float {}
impl OclDType for cl_double {}

pub struct OclTensor<T: OclDType, D: Dims> {
    buffer: Buffer<T>,
    dims: D,
    buffer_dims: D,
    dependency: RefCell<Option<Rc<Event>>>,
}

impl<T: OclDType, D: Dims> ITensor<T, D> for OclTensor<T, D> {
    #[inline]
    fn len(&self) -> usize {
        self.dims.tensor_len()
    }
    #[inline]
    fn dims(&self) -> &D {
        &self.dims
    }
}

impl<T: OclDType, D: Dims> OclTensor<T, D> {
    pub fn new(context: &Context, dims: D) -> Result<Self> {
        let buffer_dims = dims.map_each(|size, _| next_multiple(size, MATRIX_PADDING_SIZE));
        let capacity = buffer_dims.tensor_len();
        let buffer = unsafe {
            Buffer::<T>::create(context, CL_MEM_READ_WRITE, capacity, ptr::null_mut())
                .map_err(|err| Error::from_cl_err(err, "Failed to create buffer"))?
        };
        Ok(OclTensor {
            buffer,
            dims,
            buffer_dims,
            dependency: RefCell::new(None),
        })
    }

    pub fn from_slice<Q>(context: &Context, queue: Q, slice: &[T], dims: D) -> Result<Self>
    where
        Q: Deref<Target = CommandQueue>,
    {
        assert_eq!(slice.len(), dims.tensor_len());
        let mut tensor = Self::new(context, dims)?;
        tensor.write_sync(queue, slice)?;
        Ok(tensor)
    }

    pub fn from_native<N, Q>(context: &Context, queue: Q, native: &N) -> Result<Self>
    where
        Q: Deref<Target = CommandQueue>,
        N: TensorBase<T, D>,
    {
        let mut tensor = Self::new(context, *native.dims())?;
        tensor.write_sync(queue, native.as_ref())?;
        Ok(tensor)
    }

    #[inline]
    pub fn buffer_dims(&self) -> &D {
        &self.buffer_dims
    }

    fn sync(&self) -> Result<()> {
        if let Some(evt) = self.dependency.borrow_mut().take() {
            evt.wait().map_err(|err| {
                Error::from_cl_err(err, "Failed to wait for buffer dependency event")
            })?;
        };
        Ok(())
    }

    pub fn read_sync<Q>(&self, queue: Q, dst: &mut [T]) -> Result<()>
    where
        Q: Deref<Target = CommandQueue>,
    {
        assert_eq!(self.len(), dst.len());
        self.sync()?;
        // TODO: reuse some buffer
        let mut tmp: Tensor<T, D> = Tensor::filled_default(self.buffer_dims);
        let read_event = unsafe {
            queue
                .enqueue_read_buffer(&self.buffer, CL_BLOCKING, 0, tmp.as_mut(), &[])
                .map_err(|err| Error::from_cl_err(err, "Failed to enqueue buffer read"))?
        };
        if cfg!(debug_assertions) {
            let status = read_event
                .command_execution_status()
                .map_err(|err| Error::from_cl_err(err, "Failed to get command execution status"))?;
            assert_eq!(status.0, CL_COMPLETE);
        }
        copy_padded(tmp.view(), TensorViewMut::from_slice(dst, self.dims));
        Ok(())
    }

    pub fn write_sync<Q>(&mut self, queue: Q, src: &[T]) -> Result<()>
    where
        Q: Deref<Target = CommandQueue>,
    {
        assert_eq!(self.len(), src.len());
        let mut tmp: Tensor<T, D> = Tensor::filled_default(self.buffer_dims);
        copy_padded(TensorView::from_slice(src, self.dims), tmp.view_mut());
        self.sync()?;
        let write_evt = unsafe {
            queue
                .enqueue_write_buffer(&mut self.buffer, CL_BLOCKING, 0, tmp.as_ref(), &[])
                .map_err(|err| Error::from_cl_err(err, "Failed to enqueue buffer read"))?
        };
        if cfg!(debug_assertions) {
            let status = write_evt
                .command_execution_status()
                .map_err(|err| Error::from_cl_err(err, "Failed to get command execution status"))?;
            assert_eq!(status.0, CL_COMPLETE);
        }
        Ok(())
    }

    #[inline]
    pub fn buffer(&self) -> &Buffer<T> {
        &self.buffer
    }

    #[inline]
    pub fn buffer_mut(&self) -> &Buffer<T> {
        &self.buffer
    }

    pub fn get_dependency(&self) -> Option<Rc<Event>> {
        self.dependency.borrow_mut().clone()
    }

    pub fn put_dependency(&self, dep: Rc<Event>) {
        self.dependency.replace(Some(dep));
    }

    pub fn as_native<Q>(&self, queue: Q) -> Result<Tensor<T, D>>
    where
        Q: Deref<Target = CommandQueue>,
        T: Default + Clone,
    {
        let mut native = Tensor::filled_default(self.dims);
        self.read_sync(queue, native.as_mut())?;
        Ok(native)
    }
}

fn copy_padded<T: Copy, D: Dims>(src: TensorView<T, D>, mut dst: TensorViewMut<T, D>) {
    if D::N < 2 {
        let copy_len = min(dst.len(), src.len());
        unsafe {
            ptr::copy_nonoverlapping(src.as_ref().as_ptr(), dst.as_mut().as_mut_ptr(), copy_len);
        }
    } else {
        for (src_row, dst_row) in zip(src.iter_first_axis(), dst.iter_first_axis_mut()) {
            copy_padded(src_row, dst_row);
        }
    }
}
