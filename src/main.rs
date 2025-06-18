use image::open;
use spritesheet_detector::analyze_spritesheet;
use std::env;

const USAGE: &str = "Usage: spritesheet_info <path_to_image> <gap_threshold>";

fn main() {
    let path = env::args().nth(1).expect(USAGE);
    let gap_threshold = env::args()
        .nth(2)
        .expect(USAGE)
        .parse()
        .expect("Invalid gap threshold");
    let img = open(&path).expect("Failed to open image");
    let info = analyze_spritesheet(&img, gap_threshold);
    dbg!(info);
}
