use crate::util::next_multiple;
use opencl3::event::Event;
use opencl3::types::cl_event;
use std::alloc::Layout;
use std::mem::ManuallyDrop;
use std::ops::Deref;
use std::{alloc, mem, ptr, slice};

const INLINE_CAPACITY: usize = 4;
const HEAP_CAPACITY_MULTIPLE: usize = 16;

#[derive(Copy, Clone)]
struct HeapStorage {
    capacity: usize,
    ptr: *mut cl_event,
}

impl HeapStorage {
    #[inline]
    fn new(required_capacity: usize) -> Self {
        let capacity = required_capacity.max(HEAP_CAPACITY_MULTIPLE);
        HeapStorage {
            capacity,
            ptr: unsafe { alloc_storage_ptr(capacity) },
        }
    }
    #[inline]
    fn ensure_capacity(&mut self, cur_len: usize, required_capacity: usize) {
        debug_assert!(required_capacity >= cur_len);
        if required_capacity <= self.capacity {
            return;
        }
        let new_capacity = next_multiple(required_capacity, HEAP_CAPACITY_MULTIPLE);
        unsafe {
            let new_ptr = alloc_storage_ptr(new_capacity);
            ptr::copy_nonoverlapping(self.ptr, new_ptr, cur_len);
            dealloc_storage_ptr(self.ptr, self.capacity);
            self.ptr = new_ptr;
            self.capacity = new_capacity;
        }
    }
    #[inline]
    unsafe fn dealloc(&self) {
        dealloc_storage_ptr(self.ptr, self.capacity);
    }
}

union EventListStorage {
    uninit: (),
    array: [cl_event; INLINE_CAPACITY],
    heap: HeapStorage,
}

impl EventListStorage {
    #[inline]
    fn uninit() -> Self {
        Self { uninit: () }
    }
    #[inline]
    fn single(event: cl_event) -> Self {
        let mut s = Self::uninit();
        unsafe {
            *s.array.as_mut_ptr() = event;
        }
        s
    }
}

#[inline]
unsafe fn retain_event(event: cl_event) {
    opencl3::event::retain_event(event).expect("Failed to retain event");
}

#[inline]
unsafe fn retain_events(mut events: *const cl_event, len: usize) {
    let end = events.add(len);
    while events < end {
        retain_event(*events);
        events = events.offset(1);
    }
}

#[inline]
unsafe fn release_event(event: cl_event) {
    opencl3::event::release_event(event).expect("Failed to release event");
}

#[inline]
unsafe fn release_events(mut events: *const cl_event, len: usize) {
    let end = events.add(len);
    while events < end {
        release_event(*events);
        events = events.offset(1);
    }
}

#[inline]
unsafe fn dealloc_storage_ptr(ptr: *const cl_event, len: usize) {
    let layout = Layout::from_size_align_unchecked(mem::size_of::<cl_event>() * len, mem::align_of::<cl_event>());
    alloc::dealloc(ptr as *mut u8, layout);
}

#[inline]
unsafe fn alloc_storage_ptr(len: usize) -> *mut cl_event {
    let layout = Layout::from_size_align_unchecked(mem::size_of::<cl_event>() * len, mem::align_of::<cl_event>());
    alloc::alloc(layout) as *mut cl_event
}

pub struct EventList {
    len: usize,
    storage: EventListStorage,
}

impl EventList {
    #[inline]
    pub fn empty() -> Self {
        EventList {
            len: 0,
            storage: EventListStorage::uninit(),
        }
    }

    #[inline]
    pub fn from_event(event: Event) -> Self {
        let event = ManuallyDrop::new(event);
        EventList {
            len: 1,
            storage: EventListStorage::single(event.get()),
        }
    }

    #[inline]
    pub fn from_event_raw(event: cl_event) -> Self {
        unsafe {
            retain_event(event);
        }
        EventList {
            len: 1,
            storage: EventListStorage::single(event),
        }
    }

    pub fn concat<L, T>(lists: L) -> Self
    where
        L: AsRef<[T]>,
        T: Deref<Target = EventList>,
    {
        let lists = lists.as_ref();
        let total_len = lists.iter().map(|l| l.len).sum();
        let mut result = Self::empty();
        unsafe {
            let mut cur_ptr = result.reserve_uninit(total_len);
            for list in lists {
                let lptr = list.as_ptr();
                retain_events(lptr, list.len);
                ptr::copy_nonoverlapping(lptr, cur_ptr, list.len);
                cur_ptr = cur_ptr.add(list.len);
            }
            result.len = total_len;
        }
        result
    }

    #[inline]
    unsafe fn reserve_uninit(&mut self, additional: usize) -> *mut cl_event {
        let cur_len = self.len;
        let new_len = cur_len + additional;
        if self.len > INLINE_CAPACITY {
            self.storage.heap.ensure_capacity(cur_len, new_len);
            self.storage.heap.ptr.add(cur_len)
        } else if new_len > INLINE_CAPACITY {
            let heap = HeapStorage::new(new_len);
            unsafe {
                ptr::copy_nonoverlapping(self.storage.array.as_ptr(), heap.ptr, cur_len);
            }
            self.storage.heap = heap;
            self.storage.heap.ptr.add(cur_len)
        } else {
            self.storage.array.as_mut_ptr().add(cur_len)
        }
    }

    #[inline]
    unsafe fn extend_impl(&mut self, events: *const cl_event, count: usize) {
        let cur_ptr = self.reserve_uninit(count);
        ptr::copy_nonoverlapping(events, cur_ptr, count);
        self.len += count;
    }

    #[inline]
    unsafe fn free_mem(&self) {
        if self.len > INLINE_CAPACITY {
            self.storage.heap.dealloc()
        }
    }

    pub fn clear(&mut self) {
        let cur_len = self.len;
        if cur_len == 0 {
            return;
        }
        unsafe {
            if cur_len > INLINE_CAPACITY {
                release_events(self.storage.heap.ptr, cur_len);
                self.storage.heap.dealloc();
            } else {
                release_events(self.storage.array.as_ptr(), cur_len);
            }
        };
        self.len = 0;
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn push(&mut self, event: Event) {
        let event = ManuallyDrop::new(event);
        unsafe {
            self.extend_impl(&event.get(), 1);
        }
    }

    pub fn push_raw(&mut self, event: cl_event) {
        unsafe {
            retain_event(event);
            self.extend_impl(&event, 1);
        }
    }

    pub fn extend_raw(&mut self, events: &[cl_event]) {
        unsafe {
            retain_events(events.as_ptr(), events.len());
            self.extend_impl(events.as_ptr(), events.len());
        }
    }

    pub fn extend(&mut self, other: &EventList) {
        let ptr = other.as_ptr();
        unsafe {
            retain_events(ptr, other.len);
            self.extend_impl(ptr, other.len);
        }
    }

    pub fn append(&mut self, other: EventList) {
        let other = ManuallyDrop::new(other);
        unsafe {
            self.extend_impl(other.as_ptr(), other.len);
            other.free_mem();
        }
    }

    #[inline]
    fn as_ptr(&self) -> *const cl_event {
        unsafe {
            if self.len > INLINE_CAPACITY {
                self.storage.heap.ptr
            } else {
                self.storage.array.as_ptr()
            }
        }
    }

    #[inline]
    pub fn as_slice(&self) -> &[cl_event] {
        unsafe { slice::from_raw_parts(self.as_ptr(), self.len) }
    }
}

impl Drop for EventList {
    fn drop(&mut self) {
        unsafe {
            release_events(self.as_ptr(), self.len);
            self.free_mem()
        }
    }
}

impl Clone for EventList {
    fn clone(&self) -> Self {
        let mut new = Self::empty();
        new.extend_raw(self.as_ref());
        new
    }
}

impl AsRef<[cl_event]> for EventList {
    #[inline]
    fn as_ref(&self) -> &[cl_event] {
        self.as_slice()
    }
}
