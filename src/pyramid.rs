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
    build_pyramid_into(image, levels, &mut pyramid);
    pyramid
}

/// Builds the pyramid into an existing buffer, reusing each level's storage when
/// its dimensions are unchanged.
///
/// In steady state (same image size and `levels` every call) this performs no
/// heap allocation: each level's pixel buffer is overwritten in place. The
/// `pyramid` is resized to the actual number of produced levels.
pub fn build_pyramid_into(image: &GrayImage, levels: usize, pyramid: &mut Vec<GrayImage>) {
    // Level 0 is a copy of the source into the (reused) buffer.
    ensure_level(pyramid, 0, image.width(), image.height());
    pyramid[0].copy_from_slice(image.as_raw());

    let mut produced = 1;
    for level in 1..levels {
        let (prev_w, prev_h) = pyramid[level - 1].dimensions();

        // Stop when the previous level can no longer be halved.
        if prev_w < 2 || prev_h < 2 {
            break;
        }

        let (new_w, new_h) = (prev_w / 2, prev_h / 2);
        ensure_level(pyramid, level, new_w, new_h);

        // Borrow the previous (read) and current (write) levels disjointly.
        let (head, tail) = pyramid.split_at_mut(level);
        let previous_level = &head[level - 1];
        let new_image = &mut tail[0];

        for y in 0..new_h {
            for x in 0..new_w {
                let px = 2 * x;
                let py = 2 * y;

                // Average 4 pixels
                let pixel1 = previous_level.get_pixel(px, py)[0] as u32;
                let pixel2 = previous_level.get_pixel(px + 1, py)[0] as u32;
                let pixel3 = previous_level.get_pixel(px, py + 1)[0] as u32;
                let pixel4 = previous_level.get_pixel(px + 1, py + 1)[0] as u32;

                let average = ((pixel1 + pixel2 + pixel3 + pixel4) / 4) as u8;

                new_image.put_pixel(x, y, Luma([average]));
            }
        }

        produced += 1;
    }

    pyramid.truncate(produced);
}

/// Ensures `pyramid[index]` exists with the given dimensions, allocating only
/// when the slot is missing or its size changed.
fn ensure_level(pyramid: &mut Vec<GrayImage>, index: usize, width: u32, height: u32) {
    if index < pyramid.len() {
        if pyramid[index].dimensions() != (width, height) {
            pyramid[index] = ImageBuffer::new(width, height);
        }
    } else {
        pyramid.push(ImageBuffer::new(width, height));
    }
}
