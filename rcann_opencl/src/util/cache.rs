use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use opencl3::program::Program;

pub(crate) struct ProgramCache {
    programs: RefCell<HashMap<String, Rc<Program>>>,
}

impl ProgramCache {
    pub(crate) fn get(&self, key: &str) -> Option<Rc<Program>> {
        let programs = self.programs.borrow();
        match programs.get(key) {
            Some(program) => {
                Some(program.clone())
            }
            None => None
        }
    }
    pub(crate) fn insert(&self, key: String, program: Rc<Program>) {
        self.programs.borrow_mut().insert(key, program);
    }
    #[allow(unused)]
    pub(crate) fn clear(&self) {
        self.programs.borrow_mut().clear();
    }
}