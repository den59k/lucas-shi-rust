// Included inside the production detector's module in the benchmark executable
// only. These wrappers do not alter the library or expose new public APIs.
// Products, 3x3 window sums and scores run as one row-by-row pass.
pub fn tensor_scores(
    gx: &ImageBuffer<Luma<i16>, Vec<i16>>,
    gy: &ImageBuffer<Luma<i16>, Vec<i16>>,
) -> Vec<(u32, u32, f32)> {
    min_eigenvalue_scores(gx, gy)
}

pub fn nms(p: &mut Vec<(u32, u32, f32)>, w: u32, h: u32) {
    non_maximum_suppression(p, w, h);
}

pub fn quality(p: &mut Vec<(u32, u32, f32)>, q: f32) {
    filter_by_quality(p, q);
}

pub fn spacing(p: &[(u32, u32, f32)], distance: u32, w: u32, h: u32) -> Vec<(u32, u32, f32)> {
    filter_by_distance(p, distance, w, h)
}
