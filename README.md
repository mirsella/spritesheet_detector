# Spritesheet Detector

`spritesheet_detector` is a Rust library that analyzes a spritesheet image and detects its grid layout and non-empty frame count. It uses signal analysis (projection profiles and autocorrelation) to identify repeating patterns, making it robust against varying resolutions and resizing artifacts.

## Features

- **Automated Grid Detection**: Automatically determines frame dimensions, columns, and rows.
- **Smart Frame Counting**: Identifies the exact number of active frames, ignoring empty trailing cells.
- **Resizing Robustness**: Handles images resized to non-integer grid multiples (e.g., assets scaled to fit 4096px limits) using approximate harmonic reduction.
- **Small-period detection**: Scores short repeats using autocorrelation and folded projection variance.
- **High Resolution Support**: Optimized for 4K+ spritesheets using parallel processing with `rayon`.
- **Projection Analysis**: Combines Alpha and Gradient signals to detect grids even in dense tilesets or sparse animations.

## Logic Overview

The detector works in three main stages:
1. **Signal Extraction**: It calculates horizontal and vertical projection profiles of the image's alpha channel and intensity gradients.
2. **Periodicity Detection**: It analyzes factors of the image dimensions (within a small tolerance range to handle resizing) and calculates autocorrelation and Variance of Variance (VoV) scores for each potential grid size.
3. **Validation**: It verifies the grid by checking if cut lines pass through empty space or sharp edges, and then performs a final frame-by-frame alpha check to count active frames.

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
spritesheet_detector = "0.4.1"
```

### Basic Example

```rust
use image::open;
use spritesheet_detector::{analyze_spritesheet, SpritesheetInfo};

fn main() {
    // Open your spritesheet image.
    let img = open("assets/example.png").expect("Failed to open image");

    // Analyze the spritesheet.
    let info: SpritesheetInfo = analyze_spritesheet(&img);

    // Print the detected information.
    println!(
        "Sprite frame: {}x{} with {} columns and {} rows, {} valid frames.",
        info.sprite_width, info.sprite_height, info.columns, info.rows, info.frame_count
    );
}
```

## Consuming in CLI

The project includes a simple CLI tool:

```bash
cargo run -- assets/example.png
```

## Testing

The library includes an extensive test suite covering original high-resolution assets, resized production assets, and edge cases:

```bash
cargo test
```

## Single images and ambiguity

The detector uses pixels only. It does not inspect filenames or know whether an image is intended for animation. Gradients use alpha-weighted luminance, so RGB values in fully transparent pixels cannot create a grid.

Approximate periods require clearer cut lines than exact periods. A two-cell candidate with negative adjacent correlation needs an independently detected grid on the perpendicular axis. This avoids splitting some outlines in half while preserving sheets with dissimilar rows or columns.

Repeated decorations can be indistinguishable from repeated animation frames. Dashed territory edges are one example. A two-frame strip with dissimilar silhouettes can also lack enough evidence to detect. Keep explicit layout metadata for these cases; a single-image result is not proof that an asset is static.

`tests/classification.rs` covers source-image regressions, sparse animation frames, repeated UI animation, and changes to invisible RGB. Fixture origins are listed in `assets/classification/README.md`.
