use crate::error::Error;
use crate::util::Result;
use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::event::{Event, CL_COMPLETE};
use opencl3::memory::{Buffer, CL_MEM_READ_WRITE};
use opencl3::types::{cl_double, cl_float, CL_BLOCKING};
use rcann::tensor::{Dims, ITensor, Tensor, TensorBase};
use std::cell::RefCell;
use std::ops::Deref;
use std::ptr;
use std::rc::Rc;

pub trait OclDType: Copy + Default {}
impl OclDType for cl_float {}
impl OclDType for cl_double {}

pub struct OclTensor<T: OclDType, D: Dims> {
    buffer: Buffer<T>,
    dims: D,
    capacity: usize,
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
        let capacity = dims.tensor_len();
        let buffer = unsafe {
            Buffer::<T>::create(context, CL_MEM_READ_WRITE, capacity, ptr::null_mut())
                .map_err(|err| Error::from_cl_err(err, "Failed to create buffer"))?
        };
        Ok(OclTensor {
            buffer,
            dims,
            capacity,
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
        let read_event = unsafe {
            queue
                .enqueue_read_buffer(&self.buffer, CL_BLOCKING, 0, dst.as_mut(), &[])
                .map_err(|err| Error::from_cl_err(err, "Failed to enqueue buffer read"))?
        };
        if cfg!(debug_assertions) {
            let status = read_event
                .command_execution_status()
                .map_err(|err| Error::from_cl_err(err, "Failed to get command execution status"))?;
            assert_eq!(status.0, CL_COMPLETE);
        }
        Ok(())
    }

    pub fn write_sync<Q>(&mut self, queue: Q, src: &[T]) -> Result<()>
    where
        Q: Deref<Target = CommandQueue>,
    {
        assert_eq!(self.len(), src.len());
        self.sync()?;
        let write_evt = unsafe {
            queue
                .enqueue_write_buffer(&mut self.buffer, CL_BLOCKING, 0, src.as_ref(), &[])
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
        let mut native = Tensor::filled_default(*&self.dims);
        self.read_sync(queue, native.as_mut())?;
        Ok(native)
    }
}
