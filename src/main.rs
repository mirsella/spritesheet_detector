use image::open;
use spritesheet_detector::analyze_spritesheet;
use std::env;

const USAGE: &str = "Usage: spritesheet_info <path_to_image>";

fn main() {
    let path = env::args().nth(1).expect(USAGE);
    let img = open(&path).expect("Failed to open image");
    let info = analyze_spritesheet(&img);
    dbg!(info);
}
