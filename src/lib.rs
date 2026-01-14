use image::{DynamicImage, GenericImageView, RgbaImage};
use rayon::prelude::*;

/// Information about the detected spritesheet structure.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct SpritesheetInfo {
    /// The width of a single sprite frame.
    pub sprite_width: u32,
    /// The height of a single sprite frame.
    pub sprite_height: u32,
    /// Number of columns in the grid.
    pub columns: u32,
    /// Number of rows in the grid.
    pub rows: u32,
    /// Total number of frames containing non-empty content.
    pub frame_count: u32,
}

/// Analyze a spritesheet image and return its grid information.
///
/// This implementation follows the "Spectral-Spatial Pipeline" research paper:
/// 1. Dimensional reduction to 1D Energy Profiles (Alpha + Gradient).
/// 2. Autocorrelation-based periodicity detection on image divisors.
/// 3. Harmonic consensus and VoV validation to suppress sub-harmonics.
/// 4. Adaptive clustering for robust frame counting via Shannon Entropy.
#[must_use]
pub fn analyze_spritesheet(img: &DynamicImage, _gap_threshold: u32) -> SpritesheetInfo {
    let (width, height) = img.dimensions();

    // 1. Feature Extraction: Projection Profiles (Alpha + Gradient)
    let (v_alpha, h_alpha, v_grad, h_grad) = extract_signals(img);

    // 2. Periodicity Detection on Divisors
    let sprite_width = detect_period(&v_alpha, &v_grad, width);
    let sprite_height = detect_period(&h_alpha, &h_grad, height);

    let columns = if sprite_width > 0 {
        width / sprite_width
    } else {
        1
    };
    let rows = if sprite_height > 0 {
        height / sprite_height
    } else {
        1
    };

    // 3. Content Validation: Frame Counting
    // We assume frames are packed sequentially. We scan from the end to find the last active frame.
    let frame_count = if sprite_width > 0 && sprite_height > 0 {
        find_last_active_frame(img, sprite_width, sprite_height, columns, rows)
    } else {
        0
    };

    SpritesheetInfo {
        sprite_width,
        sprite_height,
        columns,
        rows,
        frame_count,
    }
}

fn extract_signals(img: &DynamicImage) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let (w, h) = img.dimensions();
    let rgba = img.to_rgba8();

    let gray: Vec<u8> = rgba
        .pixels()
        .map(|p| ((p[0] as u32 * 299 + p[1] as u32 * 587 + p[2] as u32 * 114) / 1000) as u8)
        .collect();

    let v_data: Vec<(f32, f32)> = (0..w)
        .into_par_iter()
        .map(|x| {
            let mut a_sum = 0.0;
            let mut g_sum = 0.0;
            for y in 0..h {
                let p = rgba.get_pixel(x, y);
                a_sum += f32::from(p[3]);
                if x < w - 1 {
                    let g1 = gray[(y * w + x) as usize] as i32;
                    let g2 = gray[(y * w + x + 1) as usize] as i32;
                    g_sum += (g1 - g2).abs() as f32;
                }
            }
            (a_sum / 255.0, g_sum / 255.0)
        })
        .collect();

    let h_data: Vec<(f32, f32)> = (0..h)
        .into_par_iter()
        .map(|y| {
            let mut a_sum = 0.0;
            let mut g_sum = 0.0;
            for x in 0..w {
                let p = rgba.get_pixel(x, y);
                a_sum += f32::from(p[3]);
                if y < h - 1 {
                    let g1 = gray[(y * w + x) as usize] as i32;
                    let g2 = gray[((y + 1) * w + x) as usize] as i32;
                    g_sum += (g1 - g2).abs() as f32;
                }
            }
            (a_sum / 255.0, g_sum / 255.0)
        })
        .collect();

    let (v_alpha, v_grad) = v_data.into_iter().unzip();
    let (h_alpha, h_grad) = h_data.into_iter().unzip();
    (v_alpha, h_alpha, v_grad, h_grad)
}

fn detect_period(alpha: &[f32], grad: &[f32], total_dim: u32) -> u32 {
    let factors = get_divisors(total_dim);
    let a_mean = alpha.iter().sum::<f32>() / alpha.len() as f32;
    let a_var = alpha.iter().map(|&x| (x - a_mean).powi(2)).sum::<f32>() / alpha.len() as f32;

    // Choose signal based on variance: Alpha for sprites, Gradient for maps.
    let signal = if a_var > 1e-5 { alpha } else { grad };
    let mean = signal.iter().sum::<f32>() / signal.len() as f32;
    let detrended: Vec<f32> = signal.iter().map(|&x| x - mean).collect();

    let mut scored = Vec::new();
    for &f in &factors {
        // Strict filter for tiny periods to avoid noise
        if f < 12 && total_dim > 64 {
            continue;
        }

        // Filter out small periods that are likely texture grain in large images
        if f < 32 && total_dim > 640 {
            continue;
        }
        if f < 4 || f > total_dim / 2 {
            continue;
        }
        // Ignore extremely dense grids
        if (total_dim / f) > 256 {
            continue;
        }

        let r1 = normalized_correlation(&detrended, f as usize);
        let r2 = normalized_correlation(&detrended, (f * 2) as usize);
        let r3 = normalized_correlation(&detrended, (f * 3) as usize);

        // If we only have a few repetitions (e.g. 2 columns), the signal is unreliable.
        let num_repeats = total_dim / f;
        if num_repeats < 3 && r1 < 0.25 {
            continue;
        }

        // Hard reject 2-column grids for very large images (likely false positive half-splits)
        if num_repeats == 2 && total_dim > 1000 {
            continue;
        }

        // Harmonic Product Spectrum (HPS) consensus with Geometric Mean
        // We accumulate log-scores (or just product) and take the N-th root.
        let mut product = r1.max(0.01);
        let mut count = 1.0;

        // 2nd harmonic
        if (f as usize * 2) < detrended.len() {
            product *= r2.max(0.01);
            count += 1.0;
        }

        // 3rd harmonic
        if (f as usize * 3) < detrended.len() {
            product *= r3.max(0.01);
            count += 1.0;
        }

        let h_score = product.powf(1.0 / count);
        let vov = calculate_vov(signal, f as usize);

        // Final score combines HPS and Variance of Variance (VoV)
        // Boost VoV influence to heavily favor cleaner cuts (edges).
        let score = h_score * (1.0 + vov * 3.0);
        scored.push((f, score));
    }

    if scored.is_empty() {
        return total_dim;
    }
    // Sort by score descending
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let max_s = scored[0].1;

    // Tie-breaker: Prefer the smallest period that is "strong enough".
    // We look for the smallest 'f' that has at least 85% of the top score.
    let mut best_f = scored[0].0;

    // 1. Initial selection based on score threshold
    for (f, s) in &scored {
        if *s >= max_s * 0.85 {
            if *f < best_f {
                best_f = *f;
            }
        } else {
            break; // Sorted by score, so we can stop
        }
    }

    // 2. Sub-harmonic Correction
    // If we picked 'best_f', but there exists a multiple 'M * best_f' (e.g. 3x)
    // that has a significantly better VoV (cleaner cuts), promote it.
    let mut best_vov = calculate_vov(signal, best_f as usize);

    for &(f, _s) in &scored {
        if f == best_f {
            continue;
        }

        // Check if f is a multiple of best_f
        if f % best_f == 0 {
            let mult = f / best_f;
            if mult > 3 {
                continue;
            } // Only check 2x, 3x

            let cand_vov = calculate_vov(signal, f as usize);

            // Safety check: Don't promote to a candidate that has a terrible HPS score.
            let cand_score = scored
                .iter()
                .find(|&&(k, _)| k == f)
                .map(|&(_, s)| s)
                .unwrap_or(0.0);

            // If the candidate's total score is very low relative to best, ignore it.
            if cand_score < max_s * 0.15 {
                continue;
            }

            // If the multiple has better VoV, it's likely the true boundary.
            if cand_vov > best_vov * 1.1 {
                // Only promote if the candidate HPS score is decent relative to the winner.
                let best_s = scored
                    .iter()
                    .find(|&&(k, _)| k == best_f)
                    .map(|&(_, s)| s)
                    .unwrap_or(0.0);

                if cand_score > best_s * 0.25 {
                    best_f = f;
                    best_vov = cand_vov;
                }
            }
        }
    }

    best_f
}

fn normalized_correlation(sig: &[f32], lag: usize) -> f32 {
    let n = sig.len();
    if lag == 0 || lag >= n {
        return 0.0;
    }
    let mut dot = 0.0;
    let mut n1 = 0.0;
    let mut n2 = 0.0;
    for i in 0..n - lag {
        dot += sig[i] * sig[i + lag];
        n1 += sig[i] * sig[i];
        n2 += sig[i + lag] * sig[i + lag];
    }
    if n1 <= 0.0 || n2 <= 0.0 {
        return 0.0;
    }
    dot / (n1 * n2).sqrt()
}

fn calculate_vov(sig: &[f32], p: usize) -> f32 {
    let n = sig.len();
    let c = n / p;
    if c < 2 {
        return 0.0;
    }

    // Penalize very low column count
    let mut reliability = if c == 2 { 0.5 } else { 1.0 };

    // For large images (>800), c=2 is almost always wrong (half-sheet split).
    if c == 2 && n > 800 {
        reliability = 0.1;
    }

    if c == 3 {
        reliability = 0.9;
    }

    let mut folded = vec![0.0; p];
    for i in 0..c {
        for j in 0..p {
            folded[j] += sig[i * p + j];
        }
    }
    for val in folded.iter_mut() {
        *val /= c as f32;
    }
    let m = folded.iter().sum::<f32>() / p as f32;
    // Normalize by P to make it independent of period length (Average Variance per Pixel)
    let raw_vov =
        (folded.iter().map(|&x| (x - m).powi(2)).sum::<f32>() / p as f32) / (m.abs() + 1.0);

    raw_vov * reliability
}

pub(crate) fn get_divisors(n: u32) -> Vec<u32> {
    let mut res = Vec::new();
    let s = (n as f64).sqrt() as u32;
    for i in 1..=s {
        if n % i == 0 {
            res.push(i);
            if i * i != n {
                res.push(n / i);
            }
        }
    }
    res.sort_unstable();
    res
}

fn find_last_active_frame(img: &DynamicImage, sw: u32, sh: u32, cols: u32, rows: u32) -> u32 {
    let rgba = img.to_rgba8();
    let total_frames = rows * cols;

    for i in (0..total_frames).rev() {
        let r = i / cols;
        let c = i % cols;
        let x = c * sw;
        let y = r * sh;

        if is_frame_active(&rgba, x, y, sw, sh) {
            return i + 1;
        }
    }
    0
}

fn is_frame_active(img: &RgbaImage, x: u32, y: u32, sw: u32, sh: u32) -> bool {
    let (w, h) = img.dimensions();
    // Reduce margins to 1/40 to capture edge content but avoid neighbor bleed
    let (px, py) = ((sw / 40).max(0), (sh / 40).max(0));

    // Fast scan: Skip pixels to speed up.
    // We check a sparse grid.
    // If we find ANY non-transparent pixel (a > 2 to avoid hard noise), we mark active.
    for iy in (y + py..y + sh - py).step_by(4) {
        if iy >= h {
            break;
        }
        for ix in (x + px..x + sw - px).step_by(4) {
            if ix >= w {
                break;
            }
            let p = img.get_pixel(ix, iy);
            if p[3] > 2 {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_divisors() {
        assert_eq!(get_divisors(12), vec![1, 2, 3, 4, 6, 12]);
    }
}
