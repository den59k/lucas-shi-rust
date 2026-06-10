use image::{ImageBuffer, Luma};

pub fn box_filter_3x3_in_place(image: &mut ImageBuffer<Luma<i16>, Vec<i16>>) {
    // Apply the separable 3x3 box filter: first across rows, then across columns
    box_filter_horizontal_3x3_in_place(image);
    box_filter_vertical_3x3_in_place(image);
}

fn box_filter_horizontal_3x3_in_place(image: &mut ImageBuffer<Luma<i16>, Vec<i16>>) {
    let (width, height) = (image.width() as usize, image.height() as usize);
    let data: &mut [i16] = image;

    for y in 0..height {
        let row = y * width;

        // Initialize the sum for the first pixel in the row
        let mut sum = data[row] as i32 + data[row + 1] as i32;
        let mut count = 2;

        // Process the first pixel in the row
        if width > 2 {
            sum += data[row + 2] as i32;
            count += 1;
        }
        data[row] = (sum / count) as i16;

        // Sliding window for the remaining pixels
        for x in 1..width {
            // Remove the left pixel from the sum
            if x > 1 {
                sum -= data[row + x - 2] as i32;
                count -= 1;
            }

            // Add the right pixel to the sum
            if x + 1 < width {
                sum += data[row + x + 1] as i32;
                count += 1;
            }

            // Store the result in the current pixel
            data[row + x] = (sum / count) as i16;
        }
    }
}

fn box_filter_vertical_3x3_in_place(image: &mut ImageBuffer<Luma<i16>, Vec<i16>>) {
    let (width, height) = (image.width() as usize, image.height() as usize);
    let data: &mut [i16] = image;

    for x in 0..width {
        // Initialize the sum for the first pixel in the column
        let mut sum = data[x] as i32 + data[width + x] as i32;
        let mut count = 2;

        // Process the first pixel in the column
        if height > 2 {
            sum += data[2 * width + x] as i32;
            count += 1;
        }
        data[x] = (sum / count) as i16;

        // Sliding window for the remaining pixels
        for y in 1..height {
            // Remove the top pixel from the sum
            if y > 1 {
                sum -= data[(y - 2) * width + x] as i32;
                count -= 1;
            }

            // Add the bottom pixel to the sum
            if y + 1 < height {
                sum += data[(y + 1) * width + x] as i32;
                count += 1;
            }

            // Store the result in the current pixel
            data[y * width + x] = (sum / count) as i16;
        }
    }
}
