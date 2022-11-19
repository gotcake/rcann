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
    pub fn from_cl_err<M>(err: ClError, msg: M) -> Self where M: Into<String> {
        Error::ClError {
            code: err.0,
            code_str: err.to_string(),
            msg: Some(msg.into()),
        }
    }
}

impl From<ClError> for Error {
    fn from(value: ClError) -> Self {
        Error::ClError {
            code: value.0,
            code_str: value.to_string(),
            msg: None,
        }
    }
}


