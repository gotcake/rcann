use std::cell::RefCell;
use std::ops::{Range};
use std::rc::Rc;
use rand::Rng;
use rand::rngs::StdRng;

trait Optimizer {
    fn bool(&mut self, id: &str) -> bool;
    fn f32(&mut self, id: &str, range: Range<f32>) -> f32;
    fn usize(&mut self, id: &str, range: Range<usize>) -> usize;
}

trait ParamFactory {
    type Output: Send;
    fn get_params(gen: &mut dyn Optimizer) -> Self::Output;
}

struct RandomParamSeed {
    rng: Rc<RefCell<StdRng>>,
}

struct RandomOptimizer {
    rng: StdRng,
}

impl Optimizer for RandomOptimizer {
    fn bool(&mut self, id: &str) -> bool {
        self.rng.gen()
    }
    fn f32(&mut self, id: &str, range: Range<f32>) -> f32 {
        self.rng.gen_range(range)
    }
    fn usize(&mut self, id: &str, range: Range<usize>) -> usize {
        self.rng.gen_range(range)
    }
}



