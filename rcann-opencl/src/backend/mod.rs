mod matmul;
mod other;

use crate::kernels::gemm::{GeMM, GemmKernel};
use crate::kernels::general::GeneralKernels;
use crate::kernels::mse::MSEKernel;
use crate::kernels::scoring::ScoringKernels;
use crate::kernels::softmax::Softmax;
use crate::kernels::transpose::TransposeKernel;
use crate::kernels::zero_padding::ZeroPaddingKernel;
use crate::tensor::OclTensor;
use crate::util::{self, FixedWidth2DProgramArgs, ProgramCache, Result, VecWidth};
use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::Device;
use opencl3::types::cl_float;
use rcann::backend::{Backend, TensorOps, TensorTyped};
use rcann::tensor::{Dims, DimsMore, ITensor, Tensor, TensorBase, TensorBaseMut, TensorView};
use std::fmt::Debug;

#[derive(Debug)]
#[allow(unused)]
pub struct OpenCLBackend {
    device: Device,
    context: Context,
    queue: CommandQueue,
    cache: ProgramCache,
    max_batch_size: usize,
    gemm_kernel: GemmKernel,
    gemm_kernel2: GeMM<f32>,
    transpose_kernel: TransposeKernel,
    zero_padding_kernel: ZeroPaddingKernel,
    general_kernels: GeneralKernels,
    scoring_kernels: ScoringKernels,
}

impl OpenCLBackend {
    pub fn from_default_device(max_batch_size: usize) -> Result<Self> {
        Self::from_device(util::get_default_device()?, max_batch_size)
    }

    pub fn from_device(device: Device, max_batch_size: usize) -> Result<Self> {
        let context = util::get_context(&device)?;
        let queue = util::create_queue(&context)?;
        let gemm_kernel = GemmKernel::new(&context)?;
        let gemm_kernel2 = GeMM::create(&context, 16, 16)?;
        let transpose_kernel = TransposeKernel::create(&context)?;
        let zero_padding_kernel = ZeroPaddingKernel::create(&context)?;
        let general_kernels = GeneralKernels::new(&context)?;
        let mse_kernel = MSEKernel::new(&context)?;
        let scoring_kernels = ScoringKernels::create(&context)?;
        let cache = ProgramCache::new();
        Ok(OpenCLBackend {
            device,
            context,
            queue,
            max_batch_size,
            gemm_kernel,
            gemm_kernel2,
            transpose_kernel,
            zero_padding_kernel,
            general_kernels,
            cache,
            scoring_kernels,
        })
    }
    #[inline]
    pub fn device(&self) -> &Device {
        &self.device
    }
    #[inline]
    pub fn context(&self) -> &Context {
        &self.context
    }
    #[inline]
    pub fn queue(&self) -> &CommandQueue {
        &self.queue
    }
}

impl TensorTyped for OpenCLBackend {
    type Float = cl_float;
    type TensorRef<'a, D: Dims> = &'a OclTensor<cl_float, D>;
    type Tensor<D: Dims> = OclTensor<cl_float, D>;
    type InputAdaptionBuff<D: Dims> = OclTensor<cl_float, D>;
    type OutputAdaptionBuff<D: Dims> = Tensor<Self::Float, D>;
}

impl TensorOps for OpenCLBackend {
    fn new_tensor_exact<D: Dims>(&self, dim: D) -> Self::Tensor<D> {
        OclTensor::zeroed(&self.context, &self.queue, dim).unwrap()
    }

    fn new_tensor_batch_sized<D: DimsMore>(&self, inner_dims: D) -> Self::Tensor<D::More> {
        OclTensor::zeroed(&self.context, &self.queue, inner_dims.insert_major(self.max_batch_size)).unwrap()
    }

    fn resize_tensor<D: Dims>(&self, tensor: &mut Self::Tensor<D>, dims: D) {
        tensor.resize_within_capacity(dims)
    }

    fn write_tensor<T, D>(&self, tensor: &mut Self::Tensor<D>, native_src: &T)
    where
        T: TensorBase<Self::Float, D>,
        D: Dims,
    {
        tensor.write_sync(&self.queue, native_src).unwrap();
    }

    fn read_tensor<T, D>(&self, tensor: &Self::Tensor<D>, native_dst: &mut T)
    where
        T: TensorBaseMut<Self::Float, D>,
        D: Dims,
    {
        tensor.read_sync(&self.queue, native_dst).unwrap();
    }

    fn new_input_adaption_buff<D: DimsMore>(&self, inner_dims: D) -> OclTensor<f32, D::More> {
        OclTensor::zeroed(&self.context, &self.queue, inner_dims.insert_major(self.max_batch_size)).unwrap()
    }

    fn new_output_adaption_buff<D: DimsMore>(&self, inner_dims: D) -> Tensor<f32, D::More> {
        Tensor::zeroed(inner_dims.insert_major(self.max_batch_size))
    }

    fn adapt_input<'a, D: Dims>(
        &self,
        buff: &'a mut OclTensor<f32, D>,
        input: TensorView<Self::Float, D>,
    ) -> &'a OclTensor<f32, D> {
        buff.resize_within_capacity(*input.dims());
        buff.write_sync(&self.queue, &input).unwrap();
        buff
    }

    fn adapt_output<'a, D: Dims>(
        &self,
        buff: &'a mut Tensor<Self::Float, D>,
        output: &'a OclTensor<f32, D>,
    ) -> &'a Tensor<Self::Float, D> {
        buff.resize_within_capacity(0.0, *output.dims());
        output.read_sync(&self.queue, buff).unwrap();
        buff
    }

    fn debug_tensor<D: Dims>(&self, tensor: &OclTensor<f32, D>) {
        let native_full = tensor.as_native_full_buffer(&self.queue).unwrap();
        println!(
            "{native_full:?} data_dims={} buffer_len={} capacity={}",
            tensor.dims(),
            tensor.buffer_len(),
            tensor.capacity()
        );
    }

    #[inline]
    fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }
}

impl Backend for OpenCLBackend {}
