# Spritesheet Detector

`spritesheet_detector` is a robust Rust library designed to automatically detect the grid structure (rows, columns, and frame count) of spritesheet images using signal processing techniques.

## Features

- **Signal-Based Detection**: Uses Alpha and Luminance Gradient projections to create 1D signals for precise layout analysis.
- **Robust Scoring**: Employs Harmonic Product Spectrum (HPS) and Variance of Variance (VoV) to identify the most likely grid periodicity.
- **False Positive Prevention**: Includes "Cut Line Analysis" to ensure single assets (1x1 grids) aren't accidentally split by internal patterns.
- **Sub-harmonic Correction**: Automatically detects and corrects "doubled" or "tripled" grid counts.
- **Frame Validation**: Counts active frames by scanning for non-empty content, robust against compression artifacts and faint pixels.
- **Non-Perfect Grid Support**: Detects grids even when the image has slight padding or inconsistent trailing pixels.
- **High Performance**: Parallelized signal extraction using `rayon`.

## Usage

### Library

```rust
use image::open;
use spritesheet_detector::{analyze_spritesheet, SpritesheetInfo};

fn main() {
    // Open your spritesheet image.
    let img = open("path/to/spritesheet.png").expect("Failed to open image");

    // Analyze the spritesheet.
    let info: SpritesheetInfo = analyze_spritesheet(&img);

    // Print the detected information.
    println!(
        "Sprite frame: {}x{} with {} columns and {} rows, {} valid frames.",
        info.sprite_width, info.sprite_height, info.columns, info.rows, info.frame_count
    );
}
```

### CLI

You can also use the included binary to inspect assets:

```bash
cargo run --release -- path/to/spritesheet.png
```

## How it works

Unlike simple pixel-searching algorithms, `spritesheet_detector` treats the image as a series of signals. It projects the alpha and luminance data onto 1D axes and analyzes the periodicity of these signals. This allows it to work reliably on complex assets where sprites might be touching or have varyingly transparent backgrounds.
