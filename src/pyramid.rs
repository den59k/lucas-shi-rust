use image::{GrayImage, ImageBuffer, Luma};

/// Builds a pyramid of images where each successive layer is half as large in width and height
///
/// This method just takes the average of the 4 pixels, no interpolation or anything like that
///
/// # Arguments
/// * `image` - Source image (grayscale)
/// * `levels` - Level count
///
/// # Returns
/// Vector of layers in descending order of size. First element is source image
pub fn build_pyramid(image: &GrayImage, levels: usize) -> Vec<GrayImage> {
    let mut pyramid = Vec::new();
    pyramid.push(image.clone());

    for level in 1..levels {
        let previous_level = &pyramid[level - 1];
        let (width, height) = (previous_level.width(), previous_level.height());

        // Проверяем, что изображение можно уменьшить
        if width < 2 || height < 2 {
            break;
        }

        let new_width = width / 2;
        let new_height = height / 2;

        let mut new_image = ImageBuffer::new(new_width, new_height);

        for y in 0..new_height {
            for x in 0..new_width {
                let px = 2 * x;
                let py = 2 * y;

                // Усреднение 4 пикселей
                let pixel1 = previous_level.get_pixel(px, py)[0] as u32;
                let pixel2 = previous_level.get_pixel(px + 1, py)[0] as u32;
                let pixel3 = previous_level.get_pixel(px, py + 1)[0] as u32;
                let pixel4 = previous_level.get_pixel(px + 1, py + 1)[0] as u32;

                let average = ((pixel1 + pixel2 + pixel3 + pixel4) / 4) as u8;

                new_image.put_pixel(x, y, Luma([average]));
            }
        }

        pyramid.push(new_image);
    }

    pyramid
}
