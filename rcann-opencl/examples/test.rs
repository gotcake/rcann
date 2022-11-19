use rcann_opencl::backend::OpenCLBackend;
use rcann_opencl::error::Error;

fn main() -> Result<(), Error> {
    let backend = OpenCLBackend::from_default_device()?;
    println!("{:#?}", backend);
    backend.test_naive_sgemm()
}