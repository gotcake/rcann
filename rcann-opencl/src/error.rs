use opencl3::error_codes::ClError;
use opencl3::types::cl_int;

#[derive(Debug)]
pub enum Error {
    ClError {
        code: cl_int,
        code_str: String,
        msg: Option<String>,
    },
    CreateProgramError(String),
    NoDevicesFound,
}

impl Error {
    pub fn from_cl_err<E, M>(err: E, msg: M) -> Self
    where
        E: Into<ClError>,
        M: Into<String>,
    {
        let err = err.into();
        Error::ClError {
            code: err.0,
            code_str: err.to_string(),
            msg: Some(msg.into()),
        }
    }
}