#[derive(Clone, Copy)]
pub struct RawPointerWrapper {
    pub raw: *mut f32,
}

unsafe impl Send for RawPointerWrapper {}

unsafe impl Sync for RawPointerWrapper {}