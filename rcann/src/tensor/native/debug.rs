use std::fmt::{Debug, DebugList, Formatter, Write, write};
use crate::tensor::{ITensor, Tensor, TensorBase, TensorCow, TensorView, TensorViewMut};

fn fmt_separated<I, T>(
    iter: &mut I,
    f: &mut Formatter,
    sep: &str,
    fmt: &mut impl FnMut(T, &mut Formatter) -> std::fmt::Result,
    limit: usize,
) -> std::fmt::Result
    where I: Iterator<Item=T>,
{
    let mut remaining = limit;
    let mut first = true;
    while remaining > 0 {
        let Some(el) = iter.next() else { break };
        if first {
            first = false;
        } else {
            f.write_str(sep)?;
        }
        fmt(el, f)?;
        remaining -= 1;
    }
    Ok(())
}

fn fmt_separated_max<I, T>(
    mut iter: I,
    len: usize,
    max: usize,
    f: &mut Formatter,
    sep: &str,
    fmt: &mut impl FnMut(T, &mut Formatter) -> std::fmt::Result
) -> std::fmt::Result
    where I: Iterator<Item=T> {
    if len > max {
        let limit = max / 2;
        let to_skip = len - (limit * 2);
        fmt_separated(&mut iter, f, sep, fmt, limit)?;
        f.write_str(sep)?;
        write!(f, "...({to_skip} hidden)")?;
        f.write_str(sep)?;
        iter.nth(to_skip - 1);
        fmt_separated(&mut iter, f, sep, fmt, limit)
    } else {
        fmt_separated(&mut iter, f, sep, fmt, len)
    }
}

const DEBUG_LIMIT_DIM_OUTER: usize = 5;
const DEBUG_LIMIT_DIM_INNER: usize = 10;
fn fmt_tensor_data<T>(t: TensorView<T>, f: &mut Formatter, depth: usize) -> std::fmt::Result where T: Debug {
    f.write_char('[')?;
    if t.len() > 0 {
        if t.dims().len() < 2 {
            fmt_separated_max(t.iter(), t.len(), DEBUG_LIMIT_DIM_INNER, f, ", ", &mut |el, f| Debug::fmt(el, f))?;
        } else {
            let indent = "   ".repeat(depth);
            let sep = format!(",\n{indent}   ");
            write!(f, "\n{indent}   ")?;
            fmt_separated_max(t.iter_first_axis(), t.dims().first(), DEBUG_LIMIT_DIM_OUTER, f, sep.as_str(), &mut |el, f| {
                fmt_tensor_data(el, f, depth + 1)
            })?;
            write!(f, "\n{indent}")?;
        }
    }
    f.write_char(']')
}

fn format_tensor<T>(t: TensorView<T>, f: &mut Formatter) -> std::fmt::Result where T: Debug {
    let suffix = format!(" dtype={} dims={} len={}", std::any::type_name::<T>(), t.dims(), t.len());
    fmt_tensor_data(t, f, 0)?;
    f.write_str(suffix.as_str())
}

impl<T> Debug for Tensor<T> where T: Debug {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        format_tensor(self.view(), f)
    }
}

impl<'a, T> Debug for TensorView<'a, T> where T: Debug {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        format_tensor(self.view(), f)
    }
}

impl<'a, T> Debug for TensorViewMut<'a, T> where T: Debug {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        format_tensor(self.view(), f)
    }
}

impl<'a, T> Debug for TensorCow<'a, T> where T: Debug {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        format_tensor(self.view(), f)
    }
}

#[cfg(test)]
mod test {
    use crate::tensor;
    use crate::tensor::{ITensor, Tensor};

    #[test]
    fn test_empty() {
        assert_eq!("[] dtype=i32 dims=(0) len=0", format!("{:?}", tensor![] as Tensor<i32>));
        assert_eq!("[] dtype=i32 dims=(2, 0) len=0", format!("{:?}", tensor![[],[]] as Tensor<i32>));
        assert_eq!("[] dtype=i32 dims=(1, 2, 0) len=0", format!("{:?}", tensor![[[],[]]] as Tensor<i32>));
    }

    #[test]
    fn test_small() {
        assert_eq!("[1, 2, 3, 4, 5] dtype=i32 dims=(5) len=5", format!("{:?}", tensor![1, 2, 3, 4, 5] as Tensor<i32>));
        assert_eq!("[\n   [1, 2],\n   [3, 4]\n] dtype=i32 dims=(2, 2) len=4", format!("{:?}", tensor![[1, 2],[3, 4]] as Tensor<i32>));
        assert_eq!("[\n   [\n      [1, 2],\n      [3, 4],\n      [5, 6]\n   ]\n] dtype=i32 dims=(1, 3, 2) len=6", format!("{:?}", tensor![[[1, 2],[3, 4],[5, 6]]] as Tensor<i32>));
    }

    #[test]
    fn test_large() {
        let a = Tensor::from_vec((0..200).collect(), (10, 20));
        let expected = r#"[
   [0, 1, 2, 3, 4, ...(10 hidden), 15, 16, 17, 18, 19],
   [20, 21, 22, 23, 24, ...(10 hidden), 35, 36, 37, 38, 39],
   ...(6 hidden),
   [160, 161, 162, 163, 164, ...(10 hidden), 175, 176, 177, 178, 179],
   [180, 181, 182, 183, 184, ...(10 hidden), 195, 196, 197, 198, 199]
] dtype=i32 dims=(10, 20) len=200"#;
        assert_eq!(expected, format!("{a:?}"))
    }

}