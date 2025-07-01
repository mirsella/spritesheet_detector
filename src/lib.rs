use image::{DynamicImage, GenericImageView};

/// Information about the spritesheet.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct SpritesheetInfo {
    /// The width of a single sprite frame.
    pub sprite_width: u32,
    /// The height of a single sprite frame.
    pub sprite_height: u32,
    /// Number of columns in the spritesheet grid.
    pub columns: u32,
    /// Number of rows in the spritesheet grid.
    pub rows: u32,
    /// Count of non-empty frames detected.
    pub frame_count: u32,
}

/// Analyze a spritesheet image and return its grid information.
///
/// # Arguments
///
/// * `img` - A reference to a `DynamicImage` representing the spritesheet.
/// * `gap_threshold` - Threshold for gaps between sprites.
///
/// # Returns
///
/// A [`SpritesheetInfo`] struct containing the frame dimensions, grid (columns/rows), and valid frame count.
///
/// # Details
///
/// This function assumes:
/// - If the image width is evenly divisible by its height, the sprites are square frames.
/// - Otherwise, it uses the pixel at `(0, 0)` as the margin/padding color to detect boundaries.
/// - A cell is counted as a valid frame if any pixel inside it is not equal to the margin color.
#[must_use]
pub fn analyze_spritesheet(img: &DynamicImage, gap_threshold: u32) -> SpritesheetInfo {
    let (width, height) = img.dimensions();

    // Shortcut: if width is evenly divisible by height,
    // assume single row square frames.
    if width.is_multiple_of(height) && width != height {
        let frames = width / height;
        return SpritesheetInfo {
            sprite_width: height,
            sprite_height: height,
            columns: frames,
            rows: 1,
            frame_count: frames,
        };
    }

    // Use the color at (0,0) as the margin/padding color.
    let margin_color = img.get_pixel(0, 0);

    // Find vertical boundaries: columns where every pixel equals the margin color.
    let mut vertical_boundaries: Vec<u32> = (0..width)
        .filter(|&x| (0..height).all(|y| img.get_pixel(x, y) == margin_color))
        .collect();
    if vertical_boundaries.first() != Some(&0) {
        vertical_boundaries.insert(0, 0);
    }
    if vertical_boundaries.last() != Some(&(width - 1)) {
        vertical_boundaries.push(width - 1);
    }

    // Find horizontal boundaries: rows where every pixel equals the margin color.
    let mut horizontal_boundaries: Vec<u32> = (0..height)
        .filter(|&y| (0..width).all(|x| img.get_pixel(x, y) == margin_color))
        .collect();
    if horizontal_boundaries.first() != Some(&0) {
        horizontal_boundaries.insert(0, 0);
    }
    if horizontal_boundaries.last() != Some(&(height - 1)) {
        horizontal_boundaries.push(height - 1);
    }

    // A gap must be at least a certain amount of pixels to be considered a valid cell boundary.
    #[allow(clippy::cast_possible_truncation)]
    let columns = std::cmp::max(
        vertical_boundaries
            .windows(2)
            .filter(|w| w[1] > w[0] + gap_threshold)
            .count(),
        1,
    ) as u32;
    #[allow(clippy::cast_possible_truncation)]
    let rows = std::cmp::max(
        horizontal_boundaries
            .windows(2)
            .filter(|w| w[1] > w[0] + gap_threshold)
            .count(),
        1,
    ) as u32;

    // Determine the uniform sprite dimensions.
    let sprite_width = width / columns;
    let sprite_height = height / rows;

    // Count valid frames: a cell is valid if any pixel in it is not the margin color.
    #[allow(clippy::cast_possible_truncation)]
    let frame_count = (0..rows)
        .flat_map(|row_idx| {
            (0..columns).filter(move |&col_idx| {
                let x_start = col_idx * sprite_width;
                let y_start = row_idx * sprite_height;
                let x_end = x_start + sprite_width;
                let y_end = y_start + sprite_height;
                !(y_start..y_end)
                    .all(|y| (x_start..x_end).all(|x| img.get_pixel(x, y) == margin_color))
            })
        })
        .count() as u32;

    SpritesheetInfo {
        sprite_width,
        sprite_height,
        columns,
        rows,
        frame_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbaImage;

    #[test]
    fn test_square_frames() {
        // Create an image 200x50, which implies 4 square frames of size 50x50.
        let mut img = RgbaImage::new(200, 50);
        // Fill the image with non-margin color (e.g., white).
        for pixel in img.pixels_mut() {
            *pixel = image::Rgba([255, 255, 255, 255]);
        }
        let dyn_img = DynamicImage::ImageRgba8(img);
        let info = analyze_spritesheet(&dyn_img, 40);
        assert_eq!(info.sprite_width, 50);
        assert_eq!(info.sprite_height, 50);
        assert_eq!(info.columns, 4);
        assert_eq!(info.rows, 1);
        assert_eq!(info.frame_count, 4);
    }
    #[test]
    fn test_asset_example() {
        let img = image::open("assets/example.png").unwrap();
        let info = analyze_spritesheet(&img, 40);
        assert_eq!(info.sprite_width, 193);
        assert_eq!(info.sprite_height, 155);
        assert_eq!(info.columns, 5);
        assert_eq!(info.rows, 4);
        assert_eq!(info.frame_count, 18);
    }
}
