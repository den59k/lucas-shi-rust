// Adapter for the pre-fusion detector, included only by the comparison runner.
// Each timed stage uses the historical representation without format conversion.
pub type ScorePlane = Vec<(u32, u32, f32)>;

pub fn score_values(scores: &ScorePlane) -> Vec<f32> {
    scores.iter().map(|p| p.2).collect()
}

pub fn score_plane(values: &[f32], w: u32) -> ScorePlane {
    values.iter().enumerate().map(|(i, &q)| (i as u32 % w, i as u32 / w, q)).collect()
}

pub fn tensor_scores(
    gx: &ImageBuffer<Luma<i16>, Vec<i16>>,
    gy: &ImageBuffer<Luma<i16>, Vec<i16>>,
) -> ScorePlane {
    let (mut a, mut b, mut c) = compute_gradient_products(gx, gy);
    box_sum_3x3_in_place(&mut a);
    box_sum_3x3_in_place(&mut b);
    box_sum_3x3_in_place(&mut c);
    min_eigenvalue_scores(&a, &b, &c)
}

pub fn nms(scores: &mut ScorePlane, w: u32, h: u32) -> Vec<(u32, u32, f32)> {
    non_maximum_suppression(scores, w, h);
    std::mem::take(scores)
}

// Historical production has no early quality floor; use its ordinary NMS.
pub fn floored_nms(scores: &mut ScorePlane, w: u32, h: u32, _q: f32) -> Vec<(u32, u32, f32)> {
    nms(scores, w, h)
}

pub fn quality(p: &mut Vec<(u32, u32, f32)>, q: f32) {
    filter_by_quality(p, q);
}

pub fn spacing(p: &[(u32, u32, f32)], distance: u32, w: u32, h: u32) -> Vec<(u32, u32, f32)> {
    filter_by_distance(p, distance, w, h)
}
