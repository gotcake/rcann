mod cache;
mod kernel_macros;

use crate::error::Error;
pub(crate) use cache::*;
pub(crate) use kernel_macros::*;
use opencl3::command_queue::{CommandQueue, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE};
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::Kernel;
use opencl3::program::Program;
use opencl3::types::cl_event;
use rcann::tensor::{Dim3, Dims};
use std::collections::hash_map::DefaultHasher;
use std::fmt::{Display, Formatter, Write};
use std::hash::{Hash, Hasher};
use std::mem;

#[repr(u8)]
#[derive(Debug, Hash, Eq, PartialEq, Clone, Copy)]
pub enum VecWidth {
    ONE = 1,
    TWO = 2,
    FOUR = 4,
    EIGHT = 8,
    SIXTEEN = 16,
}

impl Display for VecWidth {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        (*self as u8).fmt(f)
    }
}

impl TryFrom<u8> for VecWidth {
    type Error = Error;
    fn try_from(value: u8) -> std::result::Result<Self, Self::Error> {
        use VecWidth::*;
        match value {
            1 => Ok(ONE),
            2 => Ok(TWO),
            4 => Ok(FOUR),
            8 => Ok(EIGHT),
            16 => Ok(SIXTEEN),
            _ => Err(Error::ConversionError(format!("Invalid vector width: {value}"))),
        }
    }
}

pub(crate) fn is_valid_vec_width(usize: u8) -> bool {
    match usize {
        1 | 2 | 4 | 8 | 16 => true,
        _ => false,
    }
}


pub(crate) fn is_power_of_two(value: usize) -> bool {
    value != 0 && (value & (value - 1)) == 0
}

const MIN_BLOCK_SIZE: usize = 4;
const MAX_BLOCK_SIZE: usize = 64;
pub(crate) fn is_valid_block_size(value: usize) -> bool {
    value >= MIN_BLOCK_SIZE && value <= MAX_BLOCK_SIZE && is_power_of_two(value)
}

#[inline]
pub(crate) const fn next_multiple(n: usize, of: usize) -> usize {
    let rem = n % of;
    if rem == 0 {
        n
    } else {
        n + (of - rem)
    }
}

#[macro_export]
macro_rules! format_c_defines {
    ($($key:expr => $val:expr),* $(,)?) => {
        format!(concat!($("#define ", $key, " {}\n" ,)*), $( $val ,)*)
    };
    ($($key:ident = $val:expr),* $(,)?) => {
        format!(concat!($("#define ", stringify!($key), " {}\n" ,)*), $( $val ,)*)
    };
}
pub(crate) use format_c_defines;

#[macro_export]
macro_rules! wrap_cl_error {
    ($res: expr, $($arg:tt)*) => {
        ($res).map_err(|err| $crate::error::Error::from_cl_err(err, format!($($arg)*)))
    }
}
pub(crate) use wrap_cl_error;

pub type Result<T> = std::result::Result<T, Error>;

pub fn hash_value<T: Hash>(value: T) -> u64 {
    let mut h = DefaultHasher::new();
    value.hash(&mut h);
    h.finish()
}

pub fn get_default_device() -> Result<Device> {
    let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
        .map_err(|err| Error::from_cl_err(err, "Failed to enumerate devices"))?
        .first()
        .ok_or(Error::NoDevicesFound)?;
    Ok(Device::new(device_id))
}

pub fn get_context(device: &Device) -> Result<Context> {
    wrap_cl_error!(Context::from_device(&device), "Failed to get context")
}

pub fn create_program(context: &Context, source: &str, options: &str) -> Result<Program> {
    Program::create_and_build_from_source(context, source, options).map_err(|err| Error::CreateProgramError(err))
}

pub fn create_kernel(program: &Program, name: &str) -> Result<Kernel> {
    wrap_cl_error!(Kernel::create(&program, name), "Failed to create kernel: {name}")
}

pub fn create_queue(context: &Context) -> Result<CommandQueue> {
    wrap_cl_error!(
        CommandQueue::create_default_with_properties(context, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, 0),
        "Failed to create queue"
    )
}

#[cfg(test)]
#[allow(unused)]
pub struct TestContext {
    pub device: Device,
    pub context: Context,
    pub queue: CommandQueue,
}

#[cfg(test)]
pub fn create_test_context() -> Result<TestContext> {
    let device = get_default_device()?;
    let context = get_context(&device)?;
    let queue = create_queue(&context)?;
    Ok(TestContext { device, context, queue })
}

pub fn get_rect_region<D: Dims, T: Sized>(dims: D) -> [usize; 3] {
    let Dim3(z, y, x) = dims.as_dim3();
    return [x * mem::size_of::<T>(), y, z];
}

#[inline]
pub fn wait_for_events(events: &[cl_event]) {
    opencl3::event::wait_for_events(events).expect("Failed to wait for events");
}

#[inline]
pub fn panic_on_error<F, T>(mut f: F) -> T
where
    F: FnMut() -> Result<T>,
{
    f().unwrap()
}

#[inline]
pub const fn max_usize(a: usize, b: usize) -> usize {
    if a < b {
        b
    } else {
        a
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn test_format_defines() {
        assert_eq!("#define FOO 1\n", format_c_defines!("FOO"=>1));
        assert_eq!("#define FOO 1\n#define BAR 2\n", format_c_defines!("FOO"=>1, "BAR"=>2));
        assert_eq!(
            "#define FOO 4\n#define BAR bar\n",
            format_c_defines!("FOO"=>1+3, "BAR"=>"bar")
        );
    }
}
