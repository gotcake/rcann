use opencl3::command_queue::{CL_QUEUE_PROFILING_ENABLE, CommandQueue};
use opencl3::context::Context;
use opencl3::device::{CL_DEVICE_TYPE_GPU, Device, get_all_devices};
use opencl3::kernel::Kernel;
use opencl3::program::Program;
use crate::error::Error;

pub struct ContextWrapper {
    context: Context,
    device: Device,
}

pub type Result<T> = std::result::Result<T, Error>;

pub fn get_default_device() -> Result<Device> {
    let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
        .map_err(|err| Error::from_cl_err(err, "Failed to enumerate devices"))?
        .first()
        .ok_or(Error::NoDevicesFound)?;
    Ok(Device::new(device_id))
}

pub fn get_context(device: &Device) -> Result<Context> {
    Context::from_device(&device)
        .map_err(|err| Error::from_cl_err(err, "Failed to get context"))
}

pub fn create_program(context: &Context, source: &str) -> Result<Program> {
    Program::create_and_build_from_source(context, source, "")
        .map_err(|err| Error::CreateProgramError(err))
}

pub fn create_kernel(program: &Program, name: &str) -> Result<Kernel> {
   Kernel::create(&program, name)
       .map_err(|err| Error::from_cl_err(err, format!("Failed to create kernel: {name}")))
}

pub fn create_queue(context: &Context) -> Result<CommandQueue> {
    CommandQueue::create_default(context, CL_QUEUE_PROFILING_ENABLE)
        .map_err(|err| Error::from_cl_err(err, "Failed to create command queue"))
}