use crate::error::Error;
use crate::kernels::Kernels;
use crate::tensor::OclTensor;
use crate::util;
use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
use opencl3::context::Context;
use opencl3::device::Device;
use opencl3::types::cl_float;
use rcann::backend::{TensorOps, TensorTyped};
use rcann::tensor::{Dims, ITensor, TensorBase, TensorBaseMut};
use std::cell::RefCell;
use std::fmt::Debug;

#[derive(Debug)]
pub struct OpenCLBackend {
    device: Device,
    context: Context,
    kernels: Kernels,
    queue: RefCell<CommandQueue>, // RefCell to grant mutability with immutable self
}

impl OpenCLBackend {
    pub fn from_default_device() -> Result<Self, Error> {
        Self::from_device(util::get_default_device()?)
    }

    pub fn from_device(device: Device) -> Result<Self, Error> {
        let context = util::get_context(&device)?;
        let kernels = Kernels::create(&context)?;
        let queue = CommandQueue::create_default(&context, CL_QUEUE_PROFILING_ENABLE)
            .expect("failed to create queue");
        Ok(OpenCLBackend {
            device,
            context,
            kernels,
            queue: RefCell::new(queue),
        })
    }
}

impl TensorTyped for OpenCLBackend {
    type DType = cl_float;
    type Tensor<D: Dims> = OclTensor<cl_float, D>;
}

impl TensorOps for OpenCLBackend {
    #[inline]
    fn new_tensor<D: Dims>(&self, dim: D) -> Self::Tensor<D> {
        OclTensor::new(&self.context, dim).unwrap()
    }

    fn resize_tensor<D: Dims>(&self, tensor: &mut Self::Tensor<D>, dims: D) {
        if &dims != tensor.dims() {
            panic!("OpenCLBackend does not support Tensor resizing");
        }
    }

    fn write_tensor<T, D>(&self, tensor: &mut Self::Tensor<D>, native_src: &T)
    where
        T: TensorBase<Self::DType, D>,
        D: Dims,
    {
        tensor
            .write_sync(self.queue.borrow_mut(), native_src.as_ref())
            .unwrap();
    }

    fn read_tensor<T, D>(&self, tensor: &Self::Tensor<D>, native_dst: &mut T)
    where
        T: TensorBaseMut<Self::DType, D>,
        D: Dims,
    {
        tensor
            .read_sync(self.queue.borrow_mut(), native_dst.as_mut())
            .unwrap();
    }
}
