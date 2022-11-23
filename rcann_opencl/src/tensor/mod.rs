pub mod event_list;

use crate::kernels;
use crate::tensor::event_list::EventList;
use crate::util::{next_multiple, Result};
use crate::{util, wrap_cl_error};
use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::memory::{Buffer, CL_MEM_READ_WRITE};
use opencl3::types::{cl_double, cl_float, cl_uint, CL_BLOCKING};
use rcann::dtype::DType;
use rcann::tensor::{Dim1, Dim2, Dims, ITensor, Tensor, TensorBase, TensorBaseMut, TensorView};
use std::cell::{Ref, RefCell};
use std::ffi::c_void;
use std::mem;
use std::ops::Deref;
use std::ptr;

pub unsafe trait OclDType: DType {}
unsafe impl OclDType for cl_float {}
unsafe impl OclDType for cl_double {}
unsafe impl OclDType for cl_uint {}

pub const BLOCK_SIZE: usize = util::max_usize(
    kernels::general::constants::UNIT_WIDTH,
    util::max_usize(
        kernels::gemm::constants::TILE_SIZE,
        kernels::transpose::constants::BLOCK_SIZE,
    ),
);

pub struct OclTensor<T: OclDType, D: Dims> {
    buffer: Buffer<T>,
    capacity: usize,
    dims: D,
    buffer_dims: D,
    deps: RefCell<EventList>,
}

pub type OclTensor1<T> = OclTensor<T, Dim1>;
pub type OclTensor2<T> = OclTensor<T, Dim2>;

impl<'a, T: OclDType, D: Dims> ITensor<D> for &'a OclTensor<T, D> {
    #[inline]
    fn len(&self) -> usize {
        self.dims.tensor_len()
    }
    #[inline]
    fn dims(&self) -> &D {
        &self.dims
    }
}

impl<T: OclDType, D: Dims> ITensor<D> for OclTensor<T, D> {
    #[inline]
    fn len(&self) -> usize {
        self.dims.tensor_len()
    }
    #[inline]
    fn dims(&self) -> &D {
        &self.dims
    }
}

fn compute_buff_dims<D: Dims>(dims: &D) -> D {
    dims.map_each(|size, _| next_multiple(size, BLOCK_SIZE))
}

impl<T: OclDType, D: Dims> OclTensor<T, D> {
    pub unsafe fn uninit(context: &Context, dims: D) -> Result<Self> {
        let buffer_dims = compute_buff_dims(&dims);
        let capacity = buffer_dims.tensor_len();
        let buffer = wrap_cl_error!(
            Buffer::<T>::create(context, CL_MEM_READ_WRITE, capacity, ptr::null_mut()),
            "Failed to create buffer"
        )?;
        Ok(OclTensor {
            buffer,
            dims,
            buffer_dims,
            capacity,
            deps: RefCell::new(EventList::empty()),
        })
    }

    pub fn zeroed(context: &Context, queue: &CommandQueue, dims: D) -> Result<Self> {
        let mut tensor = unsafe { Self::uninit(context, dims)? };
        tensor.fill(queue, T::ZERO)?;
        Ok(tensor)
    }

    pub fn fill(&mut self, queue: &CommandQueue, value: T) -> Result<()> {
        let fill_event = {
            let deps = self.deps.borrow();
            wrap_cl_error!(
                unsafe {
                    queue.enqueue_fill_buffer(
                        &mut self.buffer,
                        &[value],
                        0,
                        self.capacity * mem::size_of::<T>(),
                        deps.as_slice(),
                    )
                },
                "Failed to enqueue fill buffer"
            )?
        };
        self.deps.replace(EventList::from_event(fill_event));
        Ok(())
    }

    pub fn resize_within_capacity(&mut self, dims: D) {
        if dims != self.dims {
            let buff_dims = compute_buff_dims(&dims);
            let req_cap = buff_dims.tensor_len();
            if req_cap > self.capacity {
                panic!("Buffer dims {buff_dims} for dims {dims} has required capacity of {req_cap}, but allocated capacity is {}", self.capacity)
            }
            self.dims = dims;
            self.buffer_dims = buff_dims
        }
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

    #[inline]
    pub fn buffer_len(&self) -> usize {
        self.buffer_dims.tensor_len()
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn sync(&self) {
        let mut deps = self.deps.borrow_mut();
        if !deps.is_empty() {
            util::wait_for_events(deps.as_slice());
            deps.clear();
        }
    }

    pub fn read_sync<R>(&self, queue: &CommandQueue, dst: &mut R) -> Result<()>
    where
        R: TensorBaseMut<T, D>,
    {
        assert!(D::N <= 3, "Unsupported dimensionality");
        assert_eq!(&self.dims, dst.dims(), "Mismatched tensor dimensions");
        let dst = dst.as_mut();
        if D::N <= 1 || self.dims == self.buffer_dims {
            wrap_cl_error!(
                unsafe { queue.enqueue_read_buffer(&self.buffer, CL_BLOCKING, 0, dst, self.deps.borrow().as_slice()) },
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
                        [0, 0, 0].as_ptr(),                             // buffer_origin
                        [0, 0, 0].as_ptr(),                             // host_origin
                        region.as_ptr(),                                // region
                        self.buffer_dims.minor() * mem::size_of::<T>(), // buffer_row_pitch
                        0,                                              // buffer_slice_pitch
                        0,                                              // host_row_pitch
                        0,                                              // host_slice_pitch
                        dst.as_mut_ptr() as *mut c_void,
                        self.deps.borrow().as_slice(),
                    )
                },
                "Failed to enqueue read buffer rect"
            )?;
        }
        self.clear_deps();
        Ok(())
    }

    pub fn write_sync<S>(&mut self, queue: &CommandQueue, src: &S) -> Result<()>
    where
        S: TensorBase<T, D>,
    {
        assert!(D::N <= 3, "Unsupported dimensionality");
        assert_eq!(&self.dims, src.dims(), "Mismatched tensor dimensions");
        let deps = self.deps.borrow();
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
                        [0, 0, 0].as_ptr(),                             // buffer_origin
                        [0, 0, 0].as_ptr(),                             // host_origin
                        region.as_ptr(),                                // region
                        self.buffer_dims.minor() * mem::size_of::<T>(), // buffer_row_pitch
                        0,                                              // buffer_slice_pitch
                        0,                                              // host_row_pitch
                        0,                                              // host_slice_pitch
                        src.as_ptr() as *mut c_void,
                        deps.as_slice(),
                    )
                },
                "Failed to enqueue write buffer rect"
            )?;
        }
        drop(deps);
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
    pub fn deps(&self) -> Ref<EventList> {
        self.deps.borrow()
    }

    #[inline]
    pub fn clear_deps(&self) {
        self.deps.replace(EventList::empty());
    }

    #[inline]
    pub fn set_deps(&self, deps: EventList) {
        self.deps.replace(deps);
    }

    pub fn as_native(&self, queue: &CommandQueue) -> Result<Tensor<T, D>> {
        let mut native = Tensor::zeroed(self.dims);
        self.read_sync(queue, &mut native)?;
        Ok(native)
    }

    pub fn as_native_full_buffer(&self, queue: &CommandQueue) -> Result<Tensor<T, D>> {
        let mut native = Tensor::zeroed(self.buffer_dims);
        wrap_cl_error!(
            unsafe {
                queue.enqueue_read_buffer(
                    &self.buffer,
                    CL_BLOCKING,
                    0,
                    native.as_mut(),
                    self.deps.borrow().as_slice(),
                )
            },
            "Failed to enqueue read buffer"
        )?;
        self.clear_deps();
        Ok(native)
    }
}

pub enum OclTensorRef<'a, T: OclDType, D: Dims> {
    Borrowed(&'a OclTensor<T, D>),
    Owned(OclTensor<T, D>),
}

impl<'a, T: OclDType, D: Dims> From<OclTensor<T, D>> for OclTensorRef<'a, T, D> {
    fn from(tensor: OclTensor<T, D>) -> Self {
        OclTensorRef::Owned(tensor)
    }
}

impl<'a, T: OclDType, D: Dims> From<&'a OclTensor<T, D>> for OclTensorRef<'a, T, D> {
    fn from(tensor: &'a OclTensor<T, D>) -> Self {
        OclTensorRef::Borrowed(tensor)
    }
}

impl<'a, T: OclDType, D: Dims> Deref for OclTensorRef<'a, T, D> {
    type Target = OclTensor<T, D>;
    #[inline]
    fn deref(&self) -> &Self::Target {
        use OclTensorRef::*;
        match self {
            &Borrowed(tensor) => tensor,
            Owned(tensor) => tensor,
        }
    }
}

#[cfg(test)]
mod test {
    use crate::tensor::OclTensor;
    use crate::util::{self, Result, TestContext};
    use approx::assert_abs_diff_eq;
    use rcann::tensor;
    use rcann::tensor::{Dim2, Tensor, Tensor1, Tensor2};

    #[test]
    fn test_block_size_1d() -> Result<()> {
        let TestContext { device, context, queue } = util::create_test_context()?;
        let native: Tensor1<f32> = tensor![0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.];
        let ocl = OclTensor::from_native(&context, &queue, &native)?;
        assert_abs_diff_eq!(native, ocl.as_native(&queue)?);
        Ok(())
    }

    #[test]
    fn test_non_block_size_1d() -> Result<()> {
        let TestContext { device, context, queue } = util::create_test_context()?;
        let native: Tensor1<f32> = tensor![0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.];
        let ocl = OclTensor::from_native(&context, &queue, &native)?;
        assert_abs_diff_eq!(native, ocl.as_native(&queue)?);
        Ok(())
    }

    #[test]
    fn test_block_size_2d() -> Result<()> {
        let TestContext { device, context, queue } = util::create_test_context()?;
        let m = 16;
        let n = 16;
        let native: Tensor2<f32> = Tensor::from_vec((0..(m * n)).into_iter().map(|n| n as f32).collect(), Dim2(m, n));
        let ocl = OclTensor::from_native(&context, &queue, &native)?;
        assert_abs_diff_eq!(native, ocl.as_native(&queue)?);
        Ok(())
    }

    #[test]
    fn test_non_block_size_2d() -> Result<()> {
        let TestContext { device, context, queue } = util::create_test_context()?;
        let m = 14;
        let n = 13;
        let native: Tensor2<f32> = Tensor::from_vec((0..(m * n)).into_iter().map(|n| n as f32).collect(), Dim2(m, n));
        let ocl = OclTensor::from_native(&context, &queue, &native)?;
        assert_abs_diff_eq!(native, ocl.as_native(&queue)?);
        Ok(())
    }

    #[test]
    fn test_3x2() -> Result<()> {
        let TestContext { device, context, queue } = util::create_test_context()?;
        let native: Tensor2<f32> = tensor![[1., 2., 3.], [4., 5., 6.]];
        let ocl = OclTensor::from_native(&context, &queue, &native)?;
        assert_abs_diff_eq!(native, ocl.as_native(&queue)?);
        Ok(())
    }

    #[test]
    fn test_2x3() -> Result<()> {
        let TestContext { device, context, queue } = util::create_test_context()?;
        let native: Tensor2<f32> = tensor![[1., 2.], [3., 4.], [5., 6.]];
        let ocl = OclTensor::from_native(&context, &queue, &native)?;
        assert_abs_diff_eq!(native, ocl.as_native(&queue)?);
        Ok(())
    }
}
