use std::borrow::BorrowMut;
use std::cell::RefCell;
use std::fmt::Debug;
use std::ptr;
use opencl3::command_queue::{CL_QUEUE_PROFILING_ENABLE, CL_QUEUE_PROPERTIES, CommandQueue};
use opencl3::context::Context;
use opencl3::device::Device;
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_KERNEL_READ_AND_WRITE, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE};
use opencl3::types::{CL_BLOCKING, cl_float, cl_int, CL_NON_BLOCKING};
use rcann::backend::{Backend, TensorOps, TensorTyped};
use rcann::tensor;
use rcann::tensor::{Tensor, ITensor, Dims, TensorBase, TensorBaseMut};
use std::ops::{Deref, DerefMut};
use std::rc::Rc;
use opencl3::event::Event;
use crate::error::Error;
use crate::kernels::Kernels;
use crate::tensor::OclTensor;
use crate::util;

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
        let  queue = CommandQueue::create_default(
            &context,
            CL_QUEUE_PROFILING_ENABLE,
        ).expect("failed to create queue");
        Ok(OpenCLBackend {
            device,
            context,
            kernels,
            queue: RefCell::new(queue),
        })
    }

    // https://cnugteren.github.io/tutorial/pages/page3.html
    pub fn test_naive_sgemm(&self) -> Result<(), Error> {

        let a: Tensor<f32> = tensor![
            [1., 2., 3.],
            [4., 5., 6.]
        ];

        let b: Tensor<f32> = tensor![
            [7., 8.],
            [9., 10.],
            [11., 12.]
        ];

        let mut c: Tensor<f32> = Tensor::filled_default((2, 2));

        let mut a_buff = unsafe {
            Buffer::<cl_float>::create(&self.context, CL_MEM_READ_ONLY, a.len(), ptr::null_mut()).expect("failed to create buffer a")
        };

        let mut b_buff = unsafe {
            Buffer::<cl_float>::create(&self.context, CL_MEM_READ_ONLY, b.len(), ptr::null_mut()).expect("failed to create buffer b")
        };
        let mut c_buff = unsafe {
            Buffer::<cl_float>::create(&self.context, CL_MEM_READ_WRITE, c.len(), ptr::null_mut()).expect("failed to create buffer c")
        };

        let queue = CommandQueue::create_default(
            &self.context,
            CL_QUEUE_PROFILING_ENABLE,
        ).expect("failed to create queue");

        let a_write_evt = unsafe {
            queue.enqueue_write_buffer(&mut a_buff, CL_NON_BLOCKING, 0, a.deref(), &[]).expect("failed to queue write for a")
        };
        let b_write_evt = unsafe {
            queue.enqueue_write_buffer(&mut b_buff, CL_NON_BLOCKING, 0, b.deref(), &[]).expect("failed to queue write for b")
        };
        let c_write_evt = unsafe {
            queue.enqueue_write_buffer(&mut c_buff, CL_NON_BLOCKING, 0, c.deref(), &[]).expect("failed to queue write for c")
        };

        let (size_m, size_k) = a.dims().unwrap_2d();
        let (_, size_n) = b.dims().unwrap_2d();

        let kernel_evt = unsafe {
            ExecuteKernel::new(&self.kernels.sgemm)
                .set_arg(&(size_m as cl_int))// M
                .set_arg(&(size_k as cl_int)) // K
                .set_arg(&(size_n as cl_int)) // N
                .set_arg(&1.0f32) // ALPHA
                .set_arg(&a_buff) // A
                .set_arg(&b_buff) // B
                .set_arg(&0.0f32) // BETA
                .set_arg(&c_buff) // C
                .set_local_work_sizes(&[size_m, size_n])
                .set_global_work_sizes(&[size_m, size_n])
                .set_wait_event(&a_write_evt)
                .set_wait_event(&b_write_evt)
                .set_wait_event(&c_write_evt)
                .enqueue_nd_range(&queue)
                .expect("failed to enqueue kernel")
        };

        let read_event = unsafe {
            queue.enqueue_read_buffer(&c_buff, CL_NON_BLOCKING, 0, c.deref_mut(), &[kernel_evt.get()])
                .expect("failed to enqueue read for c")
        };

        read_event.wait()?;

        println!("c = {:?}", c.deref());

        Ok(())
    }

}

impl TensorTyped for OpenCLBackend {
    type DType = cl_float;
    type Tensor = OclTensor<cl_float>;
}

impl TensorOps for OpenCLBackend {

    #[inline]
    fn new_tensor<D>(&self, dim: D) -> Self::Tensor where D: Into<Dims> {
        OclTensor::new(&self.context, dim).unwrap()
    }


    fn resize_tensor<D>(&self, tensor: &mut Self::Tensor, dims: D) where D: Into<Dims> {
        let dims = dims.into();
        if &dims != tensor.dims() {
            panic!("OpenCLBackend does not support Tensor resizing");
        }
    }

    fn write_tensor<T>(&self, tensor: &mut Self::Tensor, native_src: &T) where T: TensorBase<Self::DType> {
        tensor.write_sync(self.queue.borrow_mut(), native_src.as_ref()).unwrap();
    }

    fn read_tensor<T>(&self, tensor: &Self::Tensor, native_dst: &mut T) where T: TensorBaseMut<Self::DType> {
        tensor.read_sync(self.queue.borrow_mut(), native_dst.as_mut()).unwrap();
    }
}