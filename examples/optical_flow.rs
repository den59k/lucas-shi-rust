use image::{GrayImage, Rgba, open};
use imageproc::drawing::{draw_cross_mut, draw_line_segment_mut};
use optical_flow_lk::{
    DEFAULT_MIN_EIGEN_THRESHOLD, TrackParams, build_pyramid, calc_optical_flow_ex,
    good_features_to_track,
};

fn main() {
    let mut prev_image = open("examples/input1.png").unwrap();
    let mut next_image = open("examples/input2.png").unwrap();

    let prev_frame: GrayImage = prev_image.clone().into_luma8();
    let next_frame: GrayImage = next_image.clone().into_luma8();

    let prev_frame_pyr = build_pyramid(&prev_frame, 4);
    let next_frame_pyr = build_pyramid(&next_frame, 4);

    let mut points = good_features_to_track(&prev_frame, 0.1, 5);
    points.truncate(100);
    let prev_points: Vec<(f32, f32)> = points.iter().map(|corner| corner.position()).collect();

    let results = calc_optical_flow_ex(
        &prev_frame_pyr,
        &next_frame_pyr,
        &prev_points,
        None,
        TrackParams {
            window_size: 21,
            max_iterations: 30,
            min_eigen_threshold: DEFAULT_MIN_EIGEN_THRESHOLD,
        },
    );

    for (result, prev) in results.iter().zip(prev_points.iter()) {
        draw_line_segment_mut(&mut next_image, result.pos, *prev, Rgba([0, 255, 0, 255]));
    }

    for corner in &points {
        let (x, y) = (corner.x as i32, corner.y as i32);
        draw_cross_mut(&mut prev_image, Rgba([255, 0, 0, 255]), x, y);
    }

    prev_image
        .save("examples/output_optical_flow_prev.png")
        .unwrap();
    next_image
        .save("examples/output_optical_flow_next.png")
        .unwrap();
}
