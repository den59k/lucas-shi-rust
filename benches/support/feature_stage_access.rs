// Included inside the production detector's module in the benchmark executable
// only. These wrappers do not alter the library or expose new public APIs.
// Products, 3x3 window sums and scores run as one row-by-row pass.
pub type ScorePlane = Vec<f32>;

pub fn score_values(scores: &ScorePlane) -> Vec<f32> {
    scores.clone()
}

pub fn score_plane(values: &[f32], _w: u32) -> ScorePlane {
    values.to_vec()
}

pub fn tensor_scores(
    gx: &image::ImageBuffer<image::Luma<i16>, Vec<i16>>,
    gy: &image::ImageBuffer<image::Luma<i16>, Vec<i16>>,
) -> ScorePlane {
    let mut scores = vec![0.0; gx.as_raw().len()];
    let mut rows = TensorRows::<i32>::default();
    min_eigenvalue_scores(gx.as_raw(), gy.as_raw(), gx.width() as usize, &mut rows, &mut scores);
    scores
}

// Every positive interior local maximum, without a quality floor.
pub fn nms(scores: &mut ScorePlane, w: u32, h: u32) -> Vec<(u32, u32, f32)> {
    let (mut rows, mut maxima) = (Default::default(), Vec::new());
    local_maxima(scores, w as usize, h as usize, 0.0, &mut rows, &mut maxima);
    maxima
}

// Production NMS: derives the quality floor, then skips scores below it.
pub fn floored_nms(scores: &mut ScorePlane, w: u32, h: u32, q: f32) -> Vec<(u32, u32, f32)> {
    let floor = quality_floor(scores, w as usize, h as usize, q);
    let (mut rows, mut maxima) = (Default::default(), Vec::new());
    local_maxima(scores, w as usize, h as usize, floor, &mut rows, &mut maxima);
    maxima
}

pub fn quality(p: &mut Vec<(u32, u32, f32)>, q: f32) {
    filter_by_quality(p, q);
}

pub fn spacing(p: &[(u32, u32, f32)], distance: u32, w: u32, h: u32) -> Vec<(u32, u32, f32)> {
    let (mut occupancy, mut out) = (OccupancyGrid::default(), Vec::new());
    enforce_spacing(p, distance, w, h, &mut occupancy, &mut out);
    out
}
