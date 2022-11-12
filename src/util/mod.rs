use std::cmp::Ordering;

pub fn max_index<T: Copy + PartialOrd>(a: &[T]) -> usize {
    a.iter()
        .enumerate()
        .max_by(|&(_, &a), &(_, &b)| {
            if a < b {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        })
        .expect("expected at least one element").0
}