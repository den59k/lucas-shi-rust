use image::{open, GrayImage, Rgba};
use imageproc::drawing::{draw_cross_mut, draw_line_segment_mut};
use optical_flow_lk::{build_pyramid, calc_optical_flow, good_features_to_track};

fn main() {
    let mut prev_image = open("examples/input1.png").unwrap();
    let mut next_image = open("examples/input2.png").unwrap();

    let prev_frame: GrayImage = prev_image.clone().into_luma8();
    let next_frame: GrayImage = next_image.clone().into_luma8();

    let prev_frame_pyr = build_pyramid(&prev_frame, 4);
    let next_frame_pyr = build_pyramid(&next_frame, 4);

    let mut points = good_features_to_track(&prev_frame, 0.1, 5);
    points.truncate(100);
    let prev_points: Vec<(f32, f32)> = points.iter().map(|&x| (x.0 as f32, x.1 as f32)).collect();

    let next_points = calc_optical_flow(&prev_frame_pyr, &next_frame_pyr, &prev_points, 21, 30);

    for (prev, next) in next_points.iter().zip(prev_points.iter()) {
        draw_line_segment_mut(&mut next_image, *prev, *next, Rgba([0, 255, 0, 255]));
    }

    for &(x, y, _) in &points {
        draw_cross_mut(&mut prev_image, Rgba([255, 0, 0, 255]), x as i32, y as i32);
    }

    prev_image
        .save("examples/output_optical_flow_prev.png")
        .unwrap();
    next_image
        .save("examples/output_optical_flow_next.png")
        .unwrap();
}
