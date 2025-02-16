use image::{open, GrayImage, Rgba};
use imageproc::drawing::draw_cross_mut;
use optical_flow_lk::good_features_to_track;

fn main() {
    let mut prev_image = open("examples/input1.png").unwrap();
    let prev_frame: GrayImage = prev_image.clone().into_luma8();

    let points = good_features_to_track(&prev_frame, 0.1, 5);

    print!("{}", points.len());
    for &(x, y, _) in &points {
        draw_cross_mut(&mut prev_image, Rgba([255, 0, 0, 255]), x as i32, y as i32);
    }

    prev_image.save("examples/output_features.png").unwrap()
}
