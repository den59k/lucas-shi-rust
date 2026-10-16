// Included inside the production detector's module in the benchmark executable
// only. These wrappers do not alter the library or expose new public APIs.
pub fn products(
    gx: &ImageBuffer<Luma<i16>, Vec<i16>>,
    gy: &ImageBuffer<Luma<i16>, Vec<i16>>,
) -> GradientProduct {
    compute_gradient_products(gx, gy)
}

pub fn smooth(p: &mut GradientProduct) {
    box_sum_3x3_in_place(&mut p.0);
    box_sum_3x3_in_place(&mut p.1);
    box_sum_3x3_in_place(&mut p.2);
}

pub fn scores(p: &GradientProduct) -> Vec<(u32, u32, f32)> {
    min_eigenvalue_scores(&p.0, &p.1, &p.2)
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
