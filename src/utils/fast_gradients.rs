use image::{GrayImage, ImageBuffer, Luma};

type GradientProduct = (
    ImageBuffer<Luma<i16>, Vec<i16>>,
    ImageBuffer<Luma<i16>, Vec<i16>>,
);

pub fn compute_gradients(
    img: &GrayImage,
    kernel_x: &[i32; 9],
    kernel_y: &[i32; 9],
) -> GradientProduct {
    let (width, height) = img.dimensions();
    let mut grad_x = ImageBuffer::new(width, height);
    let mut grad_y = ImageBuffer::new(width, height);

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let mut gx: i32 = 0;
            let mut gy: i32 = 0;

            // Применяем ядра
            for ky in 0..3 {
                for kx in 0..3 {
                    let pixel = img.get_pixel(x + kx - 1, y + ky - 1)[0] as i32;
                    gx += pixel * kernel_x[(ky * 3 + kx) as usize];
                    gy += pixel * kernel_y[(ky * 3 + kx) as usize];
                }
            }

            grad_x.put_pixel(x, y, Luma([gx as i16]));
            grad_y.put_pixel(x, y, Luma([gy as i16]));
        }
    }

    (grad_x, grad_y)
}
