mod matmul;
mod other;

use crate::kernels::gemm::GeMMProgram;
use crate::kernels::scoring::ScoringProgram;
use crate::kernels::transpose::TransposeProgram;
use crate::tensor::{OclFloat, OclTensor};
use crate::util::{self, ProgramCache, Result, VecWidth};
use opencl3::command_queue::CommandQueue;
use opencl3::context::Context;
use opencl3::device::Device;
use rcann::backend::{Backend, TensorOps, TensorTyped};
use rcann::tensor::{Dims, DimsMore, ITensor, Tensor, TensorBase, TensorBaseMut, TensorView};
use std::fmt::Debug;
use crate::kernels::BUFFER_BLOCK_SIZE;
use crate::kernels::general::GeneralProgram;
use crate::kernels::zero_padding::ZeroPadProgram;

#[derive(Debug)]
#[allow(unused)]
pub struct OpenCLBackend<F: OclFloat> {
    device: Device,
    context: Context,
    queue: CommandQueue,
    cache: ProgramCache,
    max_batch_size: usize,
    vec_width: VecWidth,
    gemm_program: GeMMProgram<F>,
    transpose_program: TransposeProgram<F>,
    zero_pad_program: ZeroPadProgram<F>,
    general_program: GeneralProgram<F>,
    scoring_program: ScoringProgram<F>,
}

impl<F: OclFloat> OpenCLBackend<F> {
    pub fn from_default_device(max_batch_size: usize, vec_width: VecWidth) -> Result<Self> {
        Self::from_device(util::get_default_device()?, max_batch_size, vec_width)
    }

    pub fn from_device(device: Device, max_batch_size: usize, vec_width: VecWidth) -> Result<Self> {
        let context = util::get_context(&device)?;
        let queue = util::create_queue(&context)?;
        let gemm_program = GeMMProgram::create(&context, vec_width, BUFFER_BLOCK_SIZE)?;
        let transpose_program = TransposeProgram::create(&context, BUFFER_BLOCK_SIZE)?;
        let zero_pad_program = ZeroPadProgram::create(&context, BUFFER_BLOCK_SIZE)?;
        let general_program = GeneralProgram::create(&context, vec_width, BUFFER_BLOCK_SIZE / vec_width as usize)?;
        let scoring_program = ScoringProgram::create(&context)?;
        let cache = ProgramCache::new();
        Ok(OpenCLBackend {
            device,
            context,
            queue,
            max_batch_size,
            vec_width,
            gemm_program,
            transpose_program,
            zero_pad_program,
            general_program,
            cache,
            scoring_program,
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

impl<F: OclFloat> TensorTyped for OpenCLBackend<F> {
    type Float = F;
    type Tensor<D: Dims> = OclTensor<F, D>;
    type TensorRef<'a, D: Dims> = &'a OclTensor<F, D>;
    type InputAdaptionBuff<D: Dims> = OclTensor<F, D>;
    type OutputAdaptionBuff<D: Dims> = Tensor<F, D>;
}

impl<F: OclFloat> TensorOps for OpenCLBackend<F> {
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

    fn new_input_adaption_buff<D: DimsMore>(&self, inner_dims: D) -> OclTensor<F, D::More> {
        OclTensor::zeroed(&self.context, &self.queue, inner_dims.insert_major(self.max_batch_size)).unwrap()
    }

    fn new_output_adaption_buff<D: DimsMore>(&self, inner_dims: D) -> Tensor<F, D::More> {
        Tensor::zeroed(inner_dims.insert_major(self.max_batch_size))
    }

    fn adapt_input<'a, D: Dims>(
        &self,
        buff: &'a mut OclTensor<F, D>,
        input: TensorView<F, D>,
    ) -> &'a OclTensor<F, D> {
        buff.resize_within_capacity(*input.dims());
        buff.write_sync(&self.queue, &input).unwrap();
        buff
    }

    fn adapt_output<'a, D: Dims>(
        &self,
        buff: &'a mut Tensor<F, D>,
        output: &'a OclTensor<F, D>,
    ) -> &'a Tensor<F, D> {
        buff.resize_within_capacity(F::ZERO, *output.dims());
        output.read_sync(&self.queue, buff).unwrap();
        buff
    }

    fn debug_tensor<D: Dims>(&self, tensor: &OclTensor<F, D>) {
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

impl<F: OclFloat> Backend for OpenCLBackend<F> {}
