// Included inside the production detector's module in the benchmark executable
// only. These wrappers do not alter the library or expose new public APIs.
// Products, 3x3 window sums and scores run as one row-by-row pass.
pub fn tensor_scores(
    gx: &ImageBuffer<Luma<i16>, Vec<i16>>,
    gy: &ImageBuffer<Luma<i16>, Vec<i16>>,
) -> Vec<f32> {
    min_eigenvalue_scores(gx, gy)
}

// Every positive interior local maximum, without a quality floor.
pub fn nms(scores: &[f32], w: u32, h: u32) -> Vec<(u32, u32, f32)> {
    local_maxima(scores, w as usize, h as usize, 0.0)
}

// Production NMS: derives the quality floor, then skips scores below it.
pub fn floored_nms(scores: &[f32], w: u32, h: u32, q: f32) -> Vec<(u32, u32, f32)> {
    let floor = quality_floor(scores, w as usize, h as usize, q);
    local_maxima(scores, w as usize, h as usize, floor)
}

pub fn quality(p: &mut Vec<(u32, u32, f32)>, q: f32) {
    filter_by_quality(p, q);
}

pub fn spacing(p: &[(u32, u32, f32)], distance: u32, w: u32, h: u32) -> Vec<(u32, u32, f32)> {
    filter_by_distance(p, distance, w, h)
}
