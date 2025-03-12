use image::open;
use spritesheet_detector::analyze_spritesheet;
use std::env;

fn main() {
    let path = env::args()
        .nth(1)
        .expect("Usage: spritesheet_info <path_to_image> <gap_threshold ?>");
    let gap_threshold = env::args()
        .nth(2)
        .map(|s| s.parse().expect("Invalid gap threshold"));
    let img = open(&path).expect("Failed to open image");
    let info = analyze_spritesheet(&img, gap_threshold);
    dbg!(info);
}
