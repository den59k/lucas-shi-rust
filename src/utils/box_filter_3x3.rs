use image::{ImageBuffer, Luma};

pub fn box_filter_3x3_in_place(image: &mut ImageBuffer<Luma<i16>, Vec<i16>>) {
    // Применяем разделяемый box filter 3x3: сначала по строкам, затем по столбцам
    box_filter_horizontal_3x3_in_place(image);
    box_filter_vertical_3x3_in_place(image);
}

fn box_filter_horizontal_3x3_in_place(image: &mut ImageBuffer<Luma<i16>, Vec<i16>>) {
    let (width, height) = (image.width(), image.height());

    for y in 0..height {
        // Инициализация суммы для первого пикселя в строке
        let mut sum = image.get_pixel(0, y)[0] as i32 + image.get_pixel(1, y)[0] as i32;
        let mut count = 2;

        // Обработка первого пикселя в строке
        if width > 2 {
            sum += image.get_pixel(2, y)[0] as i32;
            count += 1;
        }
        image.put_pixel(0, y, Luma([(sum / count) as i16]));

        // Скользящее окно для остальных пикселей
        for x in 1..width {
            // Убираем левый пиксель из суммы
            if x > 1 {
                sum -= image.get_pixel(x - 2, y)[0] as i32;
                count -= 1;
            }

            // Добавляем правый пиксель в сумму
            if x + 1 < width {
                sum += image.get_pixel(x + 1, y)[0] as i32;
                count += 1;
            }

            // Сохраняем результат в текущем пикселе
            image.put_pixel(x, y, Luma([(sum / count) as i16]));
        }
    }
}

fn box_filter_vertical_3x3_in_place(image: &mut ImageBuffer<Luma<i16>, Vec<i16>>) {
    let (width, height) = (image.width(), image.height());

    for x in 0..width {
        // Инициализация суммы для первого пикселя в столбце
        let mut sum = image.get_pixel(x, 0)[0] as i32 + image.get_pixel(x, 1)[0] as i32;
        let mut count = 2;

        // Обработка первого пикселя в столбце
        if height > 2 {
            sum += image.get_pixel(x, 2)[0] as i32;
            count += 1;
        }
        image.put_pixel(x, 0, Luma([(sum / count) as i16]));

        // Скользящее окно для остальных пикселей
        for y in 1..height {
            // Убираем верхний пиксель из суммы
            if y > 1 {
                sum -= image.get_pixel(x, y - 2)[0] as i32;
                count -= 1;
            }

            // Добавляем нижний пиксель в сумму
            if y + 1 < height {
                sum += image.get_pixel(x, y + 1)[0] as i32;
                count += 1;
            }

            // Сохраняем результат в текущем пикселе
            image.put_pixel(x, y, Luma([(sum / count) as i16]));
        }
    }
}
