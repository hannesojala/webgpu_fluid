// A bad idea
pub fn to_raw<T>(s: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of::<T>() * s.len())
    }
}

// I should try to find where I stole this function from
// It was a image processing example I believe
pub fn dispatch_size(
    (width, height): (usize, usize),
    (workgroup_width, workgroup_height): (usize, usize),
) -> (u32, u32) {
    let x = (width + workgroup_width - 1) / workgroup_width;
    let y = (height + workgroup_height - 1) / workgroup_height;
    (x as u32, y as u32)
}
