use crate::error::Error;
use crate::util::{next_multiple, Result};
use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::event::{Event};
use opencl3::memory::{Buffer, CL_MEM_READ_WRITE};
use opencl3::types::{cl_double, cl_float, CL_BLOCKING, cl_event};
use rcann::tensor::{Dims, ITensor, Tensor, TensorBase, TensorBaseMut, TensorView};
use std::cell::RefCell;
use std::ffi::c_void;
use std::ptr;
use std::rc::Rc;
use crate::{util, wrap_cl_error};
use std::mem;
use crate::kernels;

pub trait OclDType: Copy + Default {}
impl OclDType for cl_float {}
impl OclDType for cl_double {}

pub const BLOCK_SIZE: usize = util::max_usize(kernels::gemm::constants::TILE_SIZE,kernels::transpose::constants::BLOCK_SIZE);

pub struct OclTensor<T: OclDType, D: Dims> {
    buffer: Buffer<T>,
    len: usize,
    capacity: usize,
    dims: D,
    buffer_dims: D,
    deps: RefCell<Vec<Rc<Event>>>,
}

impl<T: OclDType, D: Dims> ITensor<T, D> for OclTensor<T, D> {
    #[inline]
    fn len(&self) -> usize {
        self.len
    }
    #[inline]
    fn dims(&self) -> &D {
        &self.dims
    }
}

impl<T: OclDType, D: Dims> OclTensor<T, D> {
    pub unsafe fn uninit(context: &Context, dims: D) -> Result<Self> {
        let len = dims.tensor_len();
        let buffer_dims = dims.map_each(|size, _| next_multiple(size, BLOCK_SIZE));
        let capacity = buffer_dims.tensor_len();
        let buffer = wrap_cl_error!(
            Buffer::<T>::create(context, CL_MEM_READ_WRITE, capacity, ptr::null_mut()),
            "Failed to create buffer"
        )?;
        Ok(OclTensor {
            buffer,
            dims,
            buffer_dims,
            len,
            capacity,
            deps: RefCell::new(Vec::new()),
        })
    }

    pub fn zeroed(context: &Context, queue: &CommandQueue, dims: D) -> Result<Self> {
        let mut tensor = unsafe { Self::uninit(context, dims)? };
        //wrap_cl_error!(queue.enqueue_barrier_with_wait_list(), "failed to finish")?;
        tensor.fill(queue, T::default())?;
        Ok(tensor)
    }

    pub fn fill(&mut self, queue: &CommandQueue, value: T) -> Result<()> {
        let deps = self.get_deps();
        let fill_event = wrap_cl_error!(
            unsafe {
                queue.enqueue_fill_buffer(
                    &mut self.buffer,
                    &[value],
                    0,
                    self.capacity,
                    deps.as_slice()
                )
            },
            "Failed to enqueue fill buffer"
        )?;
        self.set_dep(fill_event);
        Ok(())
    }

    pub fn from_slice(context: &Context, queue: &CommandQueue, slice: &[T], dims: D) -> Result<Self> {
        Self::from_native(context, queue, &TensorView::from_slice(slice, dims))
    }

    pub fn from_native<N>(context: &Context, queue: &CommandQueue, native: &N) -> Result<Self>
    where
        N: TensorBase<T, D>,
    {
        let mut tensor = Self::zeroed(context, queue, *native.dims())?;
        tensor.write_sync(queue, native)?;
        Ok(tensor)
    }

    #[inline]
    pub fn buffer_dims(&self) -> &D {
        &self.buffer_dims
    }

    pub fn sync(&self) -> Result<()> {
        if self.has_deps() {
            let deps = self.deps.replace(Vec::new());
            util::wait_for_events(util::get_raw_events(&deps).as_slice())
        } else {
            Ok(())
        }
    }

    pub fn read_sync<R>(&self, queue: &CommandQueue, dst: &mut R) -> Result<()>
        where R: TensorBaseMut<T, D>,
    {
        assert!(D::N <= 3, "Unsupported dimensionality");
        assert_eq!(&self.dims, dst.dims(), "Mismatched tensor dimensions");
        let dst = dst.as_mut();
        if D::N <= 1 || self.dims == self.buffer_dims {
            wrap_cl_error!(
                unsafe { queue.enqueue_read_buffer(&self.buffer, CL_BLOCKING, 0, dst, self.get_deps().as_slice()) },
                "Failed to enqueue read buffer"
            )?;
        } else {
            let region = util::get_rect_region::<D, T>(self.dims);
            wrap_cl_error!(
                unsafe {
                    // see https://registry.khronos.org/OpenCL/sdk/1.1/docs/man/xhtml/clEnqueueWriteBufferRect.html
                    queue.enqueue_read_buffer_rect(
                        &self.buffer,
                        CL_BLOCKING,
                        [0, 0, 0].as_ptr(), // buffer_origin
                        [0, 0, 0].as_ptr(), // host_origin
                        region.as_ptr(), // region
                        self.buffer_dims.last() * mem::size_of::<T>(), // buffer_row_pitch
                        0, // buffer_slice_pitch
                        0, // host_row_pitch
                        0, // host_slice_pitch
                        dst.as_mut_ptr() as *mut c_void,
                        self.get_deps().as_slice()
                    )
                },
                "Failed to enqueue read buffer rect"
            )?;
        }
        self.clear_deps();
        Ok(())
    }

    pub fn write_sync<S>(&mut self, queue: &CommandQueue, src: &S) -> Result<()> where S: TensorBase<T, D> {
        assert!(D::N <= 3, "Unsupported dimensionality");
        assert_eq!(&self.dims, src.dims(), "Mismatched tensor dimensions");
        let deps = self.get_deps();
        let src = src.as_ref();
        if D::N <= 1 || self.dims == self.buffer_dims {
            wrap_cl_error!(
                unsafe { queue.enqueue_write_buffer(&mut self.buffer, CL_BLOCKING, 0, src, deps.as_slice()) },
                "Failed to enqueue write buffer"
            )?;
        } else {
            let region = util::get_rect_region::<D, T>(self.dims);
            wrap_cl_error!(
                unsafe {
                    // see https://man.opencl.org/clEnqueueWriteBufferRect.html
                    queue.enqueue_write_buffer_rect(
                        &mut self.buffer,
                        CL_BLOCKING,
                        [0, 0, 0].as_ptr(), // buffer_origin
                        [0, 0, 0].as_ptr(), // host_origin
                        region.as_ptr(), // region
                        self.buffer_dims.last() * mem::size_of::<T>(), // buffer_row_pitch
                        0, // buffer_slice_pitch
                        0, // host_row_pitch
                        0, // host_slice_pitch
                        src.as_ptr() as *mut c_void,
                        deps.as_slice()
                    )
                },
                "Failed to enqueue write buffer rect"
            )?;
        }
        self.clear_deps();
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

    #[inline]
    pub fn has_deps(&self) -> bool {
        !self.deps.borrow().is_empty()
    }

    pub fn get_deps(&self) -> Vec<cl_event> {
        util::get_raw_events(&self.deps.borrow())
    }

    pub fn set_dep<E>(&self, dep: E) where E: Into<Rc<Event>> {
        self.deps.replace(vec![dep.into()]);
    }

    pub fn set_deps<E>(&self, deps: Vec<E>) where E: Into<Rc<Event>> {
        self.deps.replace(deps.into_iter().map(|e|e.into()).collect());
    }

    fn clear_deps(&self) {
        self.deps.replace(Vec::new());
    }

    pub fn as_native(&self, queue: &CommandQueue) -> Result<Tensor<T, D>>
    where
        T: Default + Clone,
    {
        let mut native = Tensor::filled_default(self.dims);
        self.read_sync(queue, &mut native)?;
        Ok(native)
    }
}

#[cfg(test)]
mod test {
    use approx::assert_abs_diff_eq;
    use rcann::tensor;
    use rcann::tensor::{Dim2, Tensor, Tensor1, Tensor2};
    use crate::tensor::OclTensor;
    use crate::util::{self, Result, TestContext};

    #[test]
    fn test_block_size_1d() -> Result<()> {
        let TestContext { device, context, queue} = util::create_test_context()?;
        let native: Tensor1<f32> = tensor![0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.];
        let ocl = OclTensor::from_native(&context, &queue, &native)?;
        assert_abs_diff_eq!(native, ocl.as_native(&queue)?);
        Ok(())
    }

    #[test]
    fn test_non_block_size_1d() -> Result<()> {
        let TestContext { device, context, queue} = util::create_test_context()?;
        let native: Tensor1<f32> = tensor![0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.];
        let ocl = OclTensor::from_native(&context, &queue, &native)?;
        assert_abs_diff_eq!(native, ocl.as_native(&queue)?);
        Ok(())
    }

    #[test]
    fn test_block_size_2d() -> Result<()> {
        let TestContext { device, context, queue} = util::create_test_context()?;
        let m = 16;
        let n = 16;
        let native: Tensor2<f32> = Tensor::from_vec((0..(m*n)).into_iter().map(|n| n as f32).collect(), Dim2(m, n));
        let ocl = OclTensor::from_native(&context, &queue, &native)?;
        assert_abs_diff_eq!(native, ocl.as_native(&queue)?);
        Ok(())
    }

    #[test]
    fn test_non_block_size_2d() -> Result<()> {
        let TestContext { device, context, queue} = util::create_test_context()?;
        let m = 14;
        let n = 13;
        let native: Tensor2<f32> = Tensor::from_vec((0..(m*n)).into_iter().map(|n| n as f32).collect(), Dim2(m, n));
        let ocl = OclTensor::from_native(&context, &queue, &native)?;
        assert_abs_diff_eq!(native, ocl.as_native(&queue)?);
        Ok(())
    }

    #[test]
    fn test_3x2() -> Result<()> {
        let TestContext { device, context, queue} = util::create_test_context()?;
        let native: Tensor2<f32> = tensor![[1., 2., 3.],[4., 5., 6.]];
        let ocl = OclTensor::from_native(&context, &queue, &native)?;
        println!("{:?}", ocl.as_native_full_buff(&queue));
        assert_abs_diff_eq!(native, ocl.as_native(&queue)?);
        Ok(())
    }

    #[test]
    fn test_2x3() -> Result<()> {
        let TestContext { device, context, queue} = util::create_test_context()?;
        let native: Tensor2<f32> = tensor![[1., 2.], [3., 4.], [5., 6.]];
        let ocl = OclTensor::from_native(&context, &queue, &native)?;
        assert_abs_diff_eq!(native, ocl.as_native(&queue)?);
        Ok(())
    }

}
