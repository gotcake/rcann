pub(crate) static KERNEL_HEADER: &'static str = include_str!("../kernels/types.cl");

macro_rules! validate {
    ($cond:expr) => {
        if !$cond {
            return Err($crate::error::Error::ValidationError(
                concat!("Validation failed (condition: ", stringify!($cond), ")").to_string()
            ));
        }
    };
    ($cond:expr, $msg: literal) => {
        if !$cond {
            return Err($crate::error::Error::ValidationError(
                concat!("Validation failed: ", $msg, " (condition: ", stringify!($cond), ")").to_string()
            ));
        }
    };
}
pub(crate) use validate;

macro_rules! let_param_assignment {
    ($getter:ident) => {};
    ($getter:ident, $name:ident) => {
        let $name = self.$getter();
    }
}
pub(crate) use let_param_assignment;

macro_rules! zero_or_more_expr {
    ($($zero:expr)?, $($more:expr)?, ) => { $($zero)? };
    ($($zero:expr)?, $($more:expr)?, $($t:tt)+) => { $($more)? };
}
pub(crate) use zero_or_more_expr;

macro_rules! one_or_more_expr {
    ($one:expr, $more:expr, $t:tt ) => { $one };
    ($one:expr, $more:expr, $($t:tt)+) => { $more };
}
pub(crate) use one_or_more_expr;

macro_rules! first_ident {
    ($first: ident $(,)?) => {
        $first
    };
    ($first: ident, $($rest:ident),+ $(,)?) => {
        $first
    }
}
pub(crate) use first_ident;

macro_rules! push_c_defines {
    // empty base case
    (
        @inner $str:ident,
        [],
        [],
    ) => {};
    // no arguments base case
    (
        @inner $str:ident,
        [$( $part:expr ),+ $(,)?],
        [],
    ) => {
        $str.push_str(concat!($($part),*), $($arg),*);
    };
    // default base case
    (
        @inner $str:ident,
        [$( $part:expr ),* $(,)?],
        [$( $arg:expr ),* $(,)?],
    ) => {
        write!($str, concat!($($part),*), $($arg),*).unwrap()
    };
    // literal case
    (
        @inner $str:ident,
        [$( $part:expr ),* $(,)?],
        [$( $arg:expr ),* $(,)?],
        $key1:ident = $val1:literal,
        $( $key:ident = $val:expr, )*
    ) => {
        push_c_defines!(
            @inner $str,
            [$($part,)* "#define ", stringify!($key1), " ", $val1, "\n"],
            [$($arg),*],
            $( $key = $val, )*
        )
    };
    // expression case
    (
        @inner $str:ident,
        [$( $part:expr ),* $(,)?],
        [$( $arg:expr ),* $(,)?],
        $key1:ident = $val1:expr,
        $( $key:ident = $val:expr, )*
    ) => {
        push_c_defines!(
            @inner $str,
            [$($part,)* "#define ", stringify!($key1), " {}\n"],
            [$($arg,)* $val1],
            $( $key = $val, )*
        )
    };
    ($str:ident, $($key:ident = $val:expr),* $(,)?) => {
        push_c_defines!(
            @inner $str,
            [],
            [],
            $( $key = $val, )*
        )
    };
}
pub (crate) use push_c_defines;

macro_rules! ocl_program {
    (
        name = $type_name:ident,
        source = $source_file:literal,
        $(generic_args = <$( $gen_ident:ident $(: $gen_constraint:tt)? ),+>,)?
        compile_param = $compile_param_name:ident : $compile_param_ty:ty,
        $(runtime_param = $runtime_param_name:ident : $runtime_param_ty:ty,)?
        $(validation = $program_validation:block,)?
        $(defines = { $($defines:tt)* },)?
        kernels = { $( $kernel_name:ident { $($kernels_tt:tt)+ }, )+ },
    ) => {

        #[allow(unused)]
        #[derive(Debug)]
        pub(crate) struct $type_name$(<$( $gen_ident $(: $gen_constraint)? ),+>)? {
            program: std::rc::Rc<opencl3::program::Program>,
            $compile_param_name: $compile_param_ty,
            $($runtime_param_name: $runtime_param_ty,)?
            $(
            $kernel_name: opencl3::kernel::Kernel,
            )+
            $(_marker: std::marker::PhantomData<$( $gen_ident ),+>)?,
        }

        #[allow(unused)]
        impl$(<$( $gen_ident $(: $gen_constraint)? ),+>)? $type_name$(<$( $gen_ident ),+>)? {

            fn compile_program(
                context: &opencl3::context::Context,
                $compile_param_name: &$compile_param_ty,
            ) -> $crate::util::Result<std::rc::Rc<opencl3::program::Program>> {
                use $crate::util::*;
                use std::fmt::Write;
                $( $program_validation; )?
                let mut code = String::new();
                $(push_c_defines!(code, $( $defines )*);)?
                code.push_str(KERNEL_HEADER);
                code.push_str(include_str!($source_file));
                Ok(std::rc::Rc::new(
                    create_program(context, code.as_str(), "")?
                ))
            }

            fn get_or_compile_program(
                context: &opencl3::context::Context,
                cache: &$crate::util::ProgramCache,
                $compile_param_name: &$compile_param_ty,
            ) -> $crate::util::Result<std::rc::Rc<opencl3::program::Program>> {
                let key = format!(concat!(stringify!($type_name), ":{}"), $crate::util::hash_value(&$compile_param_name));
                Ok(match cache.get(key.as_str()) {
                    Some(program) => program,
                    None => {
                        let program = Self::compile_program(context, &$compile_param_name)?;
                        cache.insert(key, program.clone());
                        program
                    }
                })
            }

            fn new(
                program: std::rc::Rc<opencl3::program::Program>,
                $compile_param_name: $compile_param_ty,
                $($runtime_param_name: $runtime_param_ty,)?
            ) -> $crate::util::Result<Self> {
                $(
                let $kernel_name = $crate::util::create_kernel(program.as_ref(), stringify!($kernel_name))?;
                )+
                Ok(Self {
                    program,
                    $compile_param_name,
                    $($runtime_param_name,)?
                    $(
                    $kernel_name,
                    )+
                    _marker: std::marker::PhantomData,
                })
            }

            pub fn create(
                context: &opencl3::context::Context,
                $compile_param_name: $compile_param_ty,
                $($runtime_param_name: $runtime_param_ty,)?
            ) -> $crate::util::Result<Self> {
                Self::new(
                    Self::compile_program(context, &$compile_param_name)?,
                    $compile_param_name,
                    $($runtime_param_name,)?
                )
            }

            pub fn get_or_create(
                context: &opencl3::context::Context,
                cache: &$crate::util::ProgramCache,
                $compile_param_name: $compile_param_ty,
                $($runtime_param_name: $runtime_param_ty,)?
            ) -> $crate::util::Result<Self> {
                Self::new(
                    Self::get_or_compile_program(context, cache, &$compile_param_name)?,
                    $compile_param_name,
                    $($runtime_param_name,)?
                )
            }

            ocl_program!(
                @impl_kernel_fn
                compile_param = $compile_param_name,
                $(runtime_param = $runtime_param_name,)?
                @kernels
                $($kernel_name { $($kernels_tt)+ },)+
            );

            pub(crate) fn $compile_param_name(&self) -> &$compile_param_ty {
                &self.$compile_param_name
            }

            $(
            pub(crate) fn $runtime_param_name(&self) -> &$runtime_param_ty {
                &self.$runtime_param_name
            }
            )?

        }

    };


    (
        @impl_kernel_fn
        $(compile_param = $compile_param_name: ident,)?
        $(runtime_param = $runtime_param_name: ident,)?
        @kernels
    ) => {};

    (
        @impl_kernel_fn
        $(compile_param = $compile_param_name: ident,)?
        $(runtime_param = $runtime_param_name: ident,)?
        @kernels
        $kernel_name:ident {
            call_params = ($( $param:ident : $param_ty:ty ),+ $(,)?),
            inputs = [$( $input:ident ),+ $(,)?],
            outputs = [$( $output:ident ),+ $(,)?],
            $(extra_args = [$( $arg:expr ),* $(,)?],)?
            global_dims = [$( $global_dim:expr ),+ $(,)?],
            $(local_dims = [$( $local_dim:expr ),* $(,)?],)?
            $(validation = $validation:block,)?
        },
        $($rest:tt)*
    ) => {
        pub(crate) fn $kernel_name(
            &self,
            queue: &opencl3::command_queue::CommandQueue,
            $(
            $param: $param_ty,
            )+
        ) {
            use opencl3::kernel::ExecuteKernel;
            use $crate::tensor::event_list::EventList;
            use rcann::tensor::*;
            use $crate::util::*;
            $(let $compile_param_name = &self.$compile_param_name;)?
            $(let $runtime_param_name = self.$runtime_param_name;)?
            $( $validation; )?
            let mut exec = ExecuteKernel::new(&self.$kernel_name);
            unsafe {
                $($(
                exec.set_arg($arg);
                )*)?
                $(
                exec.set_arg($input.buffer());
                )+
                $(
                exec.set_arg($output.buffer());
                )+
            }
            exec.set_global_work_sizes(&[$( $global_dim ),*]);
            $(zero_or_more_expr!(, exec.set_local_work_sizes(&[$( $local_dim ),*]), $( $local_dim )*);)?
            let deps = one_or_more_expr!(
                $( $input.deps() )+,
                EventList::concat([$( $input.deps() ),+]),
                $( $input )+
            );
            exec.set_event_wait_list(deps.as_slice());
            let event = EventList::from_event(unsafe {
                exec.enqueue_nd_range(queue)
                    .expect(concat!("Failed to enqueue ", stringify!($kernel_name), " kernel"))
            });
            $(
            $output.set_deps(event.clone());
            )+
        }
        ocl_program!(
            @impl_kernel_fn
            $(compile_param = $compile_param_name,)?
            $(runtime_param = $runtime_param_name,)?
            @kernels
            $( $rest )*
        );
    };

}
pub(crate) use ocl_program;

macro_rules! row_based_ocl_program {
    (
        name = $type_name:ident,
        source = $source_file:literal,
        kernels = {
            $($kernel_name:ident {
                call_params = ($( $param:ident : $param_ty:ty ),+ $(,)?),
                inputs = [$( $input:ident ),+ $(,)?],
                outputs = [$( $output:ident ),+ $(,)?],
                $(extra_args = [$( $arg:expr ),* $(,)?],)?
                global_dims = [$( $global_dim:expr ),+ $(,)?],
                $(local_dims = [$( $local_dim:expr ),* $(,)?],)?
                $(validation = $validation:block,)?
            },)+
        },
    ) => {
        ocl_program!(
            name = $type_name,
            source = $source_file,
            generic_args = <T: OclFloat>,
            compile_param = args: $crate::util::FixedWidth2DProgramArgs,
            validation = {
                validate!(args.cols > 0);
                validate!(args.cols <= args.row_stride);
                validate!(args.row_stride % args.vec_width as usize == 0);
            },
            defines = {
                FLOAT_BITS = T::BITS,
                VEC_WIDTH = args.vec_width,
                VEC_COLS = args.cols / args.vec_width as usize,
                VEC_COLS_REM = args.cols % args.vec_width as usize,
                COLS = args.cols,
                ROW_STRIDE = args.row_stride,
            },
            kernels = {
                $($kernel_name {
                    call_params = ($( $param : $param_ty ),+),
                    inputs = [$( $input ),+],
                    outputs = [$( $output ),+],
                    extra_args = [
                        &(first_ident!($( $input ),+).dims().rows() as u32),
                        $($( $arg ),*)?
                    ],
                    global_dims = [$( $global_dim ),+],
                    $(local_dims = [$( $local_dim ),*],)?
                    validation = {
                        let rows = first_ident!($( $input ),+).dims().rows();
                        $(
                        assert_eq!($input.dims(), &Dim2(rows, args.cols));
                        assert_eq!($input.buffer_dims().cols(), args.row_stride);
                        )+
                        $(
                        assert_eq!($output.dims(), &Dim2(rows, args.cols));
                        assert_eq!($output.buffer_dims().cols(), args.row_stride);
                        )+
                        $( $validation )?
                    },
                },)+
            },
        );
    };
}
pub(crate) use row_based_ocl_program;
