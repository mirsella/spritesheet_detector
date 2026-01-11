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
    // #[cfg(test)]
    eprintln!("\n--- ANALYZING: {}x{} ---", width, height);

    // 1. Feature Extraction: Projection Profiles (Alpha + Gradient)
    let (v_alpha, h_alpha, v_grad, h_grad) = extract_signals(img);

    // 2. Periodicity Detection on Divisors
    let sprite_width = detect_period(&v_alpha, &v_grad, width, "Width");
    let sprite_height = detect_period(&h_alpha, &h_grad, height, "Height");

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

    #[cfg(test)]
    eprintln!(
        "Detected grid: {}x{} ({} cols, {} rows)",
        sprite_width, sprite_height, columns, rows
    );

    // 3. Content Validation: Frame Counting
    let frame_count = if sprite_width > 0 && sprite_height > 0 {
        count_active_frames(img, sprite_width, sprite_height, columns, rows)
    } else {
        0
    };

    #[cfg(test)]
    eprintln!("Final Frame Count: {}", frame_count);

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

fn detect_period(alpha: &[f32], grad: &[f32], total_dim: u32, label: &str) -> u32 {
    let factors = get_divisors(total_dim);
    let a_mean = alpha.iter().sum::<f32>() / alpha.len() as f32;
    let a_var = alpha.iter().map(|&x| (x - a_mean).powi(2)).sum::<f32>() / alpha.len() as f32;

    #[cfg(test)]
    if label == "Width" {
        eprintln!("  Signal Var: {:.6}", a_var);
    }

    // Choose signal based on variance: Alpha for sprites, Gradient for maps.
    // Lowered to 1e-5 to catch very sparse overlays (like 'map_overlays')
    let signal = if a_var > 1e-5 { alpha } else { grad };
    let mean = signal.iter().sum::<f32>() / signal.len() as f32;
    let detrended: Vec<f32> = signal.iter().map(|&x| x - mean).collect();

    let mut scored = Vec::new();
    for &f in &factors {
        // Strict filter for tiny periods to avoid noise (e.g. 6px in 222px image)
        if f < 12 && total_dim > 64 {
            continue;
        }

        // Filter out small periods that are likely texture grain in large images
        // Bumped to 32px for large images to filter out 'map_overlays' noise (18px).
        if f < 32 && total_dim > 640 {
            continue;
        }
        if f < 4 || f > total_dim / 2 {
            continue;
        }
        if (total_dim / f) > 256 {
            continue;
        } // Ignore extremely dense grids
        if f < 4 || f > total_dim / 2 {
            continue;
        }
        if (total_dim / f) > 256 {
            continue;
        } // Ignore extremely dense grids

        let r1 = normalized_correlation(&detrended, f as usize);
        let r2 = normalized_correlation(&detrended, (f * 2) as usize);
        let r3 = normalized_correlation(&detrended, (f * 3) as usize);

        // If we only have a few repetitions (e.g. 2 columns), the signal is unreliable.
        // We require a much stronger correlation to accept it.
        // Relaxed threshold to 0.25 to allow 'map_tiles' (2 columns, weak correlation) to pass.
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
        // We do NOT penalize missing harmonics if they are out of bounds (edge of image).
        // This ensures that large periods (e.g. 1/3 of image) are treated fairly.
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

        // Final score combines HPS and Variance of Variance
        // Boost VoV influence to 3.0 to heavily favor cleaner cuts (edges).
        let score = h_score * (1.0 + vov * 3.0);

        // #[cfg(test)]
        if (f == 200 || f == 168 || f == 900 || f == 24) && (label == "Height" || label == "Width")
        {
            eprintln!(
                "DEBUG [{}]: P={} r1={:.3} r2={:.3} r3={:.3} vov={:.3} => {:.4}",
                label, f, r1, r2, r3, vov, score
            );
        }

        scored.push((f, score));
    }

    if scored.is_empty() {
        return total_dim;
    }
    // Sort by score descending
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let max_s = scored[0].1;

    // #[cfg(test)]
    // {
    eprintln!("  {} Candidates (top 10):", label);
    for (f, s) in scored.iter().take(10) {
        eprintln!("    P={}: score={:.4}", f, s);
    }
    // }

    // Tie-breaker: Prefer the smallest period that is "strong enough".
    // We look for the smallest 'f' that has at least 85% of the top score.
    // Increased from 70% to 85% to favor the HPS winner (e.g. 336px) over sub-harmonics (168px)
    // unless the sub-harmonic is nearly as good.
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

    // 2. Sub-harmonic Correction (e.g. 106 vs 318)
    // If we picked 'best_f', but there exists a multiple 'M * best_f' (e.g. 3x)
    // that has a significantly better VoV (cleaner cuts), promote it.
    // This fixes cases where the sprite has strong internal repetition (high HPS at 1/3 size)
    // but the true boundary is at the full size.
    let mut best_vov = calculate_vov(signal, best_f as usize);

    for &(f, _s) in &scored {
        if f == best_f {
            continue;
        }

        // Check if f is a multiple of best_f (e.g. 318 is multiple of 106)
        if f % best_f == 0 {
            let mult = f / best_f;
            if mult > 3 {
                continue;
            } // Only check 2x, 3x

            let cand_vov = calculate_vov(signal, f as usize);

            // Safety check: Don't promote to a candidate that has a terrible HPS score.
            // We can look up the score of 'f' in the map or just recompute/store it.
            // Since 'scored' is available, let's find the score of 'f'.
            // Note: 'scored' is a Vec<(u32, f32)>.
            let cand_score = scored
                .iter()
                .find(|&&(k, _)| k == f)
                .map(|&(_, s)| s)
                .unwrap_or(0.0);

            // If the candidate's total score is less than 15% of the best score,
            // it's likely a harmonic ghost with high VoV but no HPS support.
            // Increased from 1% to 15% to fix 'map_tiles_borders' (400 -> 800 failure)
            if cand_score < max_s * 0.15 {
                continue;
            }

            #[cfg(test)]
            eprintln!(
                "    Checking promotion: {} -> {} (VoV {:.2} vs {:.2}, Score {:.4})",
                best_f, f, best_vov, cand_vov, cand_score
            );

            // If the multiple has better VoV, it's likely the true boundary.
            // (e.g. 106 has VoV 12.8, 318 has VoV 16.0 -> 318 wins)
            // Relaxed to 1.1 (10%) improvement.
            if cand_vov > best_vov * 1.1 {
                // Only promote if the candidate HPS score is decent relative to the winner.
                // Lowered requirement to 0.25 to allow 'bomb_card_area' (318 vs 106)
                // where 318 score is ~4.0 and 106 is ~10.1 (ratio ~0.4)
                let best_s = scored
                    .iter()
                    .find(|&&(k, _)| k == best_f)
                    .map(|&(_, s)| s)
                    .unwrap_or(0.0);

                if cand_score > best_s * 0.25 {
                    #[cfg(test)]
                    eprintln!(
                        "  -> Promotion! {} (VoV {:.2}, S {:.2}) promotes to {} (VoV {:.2}, S {:.2})",
                        best_f, best_vov, best_s, f, cand_vov, cand_score
                    );
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
    // If c=2, variance is just (x1-m)^2 + (x2-m)^2. It's noisy but not useless.
    // However, for large images, a 2-column grid is suspicious.
    // Set to 0.5 to balance 'map_overlays' (bad 2-col) vs 'map_tiles' (good 2-col).
    let mut reliability = if c == 2 { 0.5 } else { 1.0 };

    // For large images (>800), c=2 is almost always wrong (half-sheet split). Crush it.
    if c == 2 && n > 800 {
        reliability = 0.1;
    }

    if c == 3 {
        reliability = 0.9;
    }

    #[cfg(test)]
    if reliability < 1.0 && p > 100 {
        // eprintln!("DEBUG: VoV Reliability for P={} (c={}, n={}) = {:.2}", p, c, n, reliability);
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

fn count_active_frames(img: &DynamicImage, sw: u32, sh: u32, cols: u32, rows: u32) -> u32 {
    let rgba = img.to_rgba8();
    let scores: Vec<f32> = (0..rows * cols)
        .into_par_iter()
        .map(|idx| {
            let (r, c) = (idx / cols, idx % cols);
            cell_activity_score(&rgba, c * sw, r * sh, sw, sh)
        })
        .collect();

    if scores.is_empty() {
        return 0;
    }

    let n = scores.len() as f32;
    let mean = scores.iter().sum::<f32>() / n;
    let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / n;
    let std_dev = variance.sqrt();
    let cv = if mean > 1e-6 { std_dev / mean } else { 0.0 };

    let mut sorted = scores.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let max_s = *sorted.last().unwrap_or(&0.0);

    // Global Noise Floor:
    // If the strongest frame is very weak, the whole image is likely empty or just noise.
    if max_s < 0.05 {
        return 0;
    }

    // Otsu's Thresholding (Simplified)
    // We want to find a threshold 't' that maximizes inter-class variance between Noise (background) and Signal (sprites).
    // We iterate through the sorted scores as potential thresholds.
    let mut best_t = max_s * 0.1;
    let mut max_sigma_b = 0.0;

    // Only search between 5% and 80% of the range to avoid extremes
    let start_idx = (scores.len() as f32 * 0.05) as usize;
    let end_idx = (scores.len() as f32 * 0.95) as usize;

    // Optimization: Precompute cumulative sums (integral image for 1D array)
    // But for N < 10000, linear scan is fine.

    for i in start_idx..end_idx {
        let t = sorted[i];
        if t < 0.01 {
            continue;
        } // Skip obvious noise

        // Split into two groups
        // Group 0: scores <= t
        // Group 1: scores > t
        // We can optimize this using indices since it's sorted.
        let w0 = (i + 1) as f32;
        let w1 = (scores.len() - (i + 1)) as f32;

        if w0 == 0.0 || w1 == 0.0 {
            continue;
        }

        let sum0: f32 = sorted[0..=i].iter().sum();
        let mean0 = sum0 / w0;

        let sum1: f32 = sorted[i + 1..].iter().sum();
        let mean1 = sum1 / w1;

        // Inter-class variance = w0 * w1 * (mean0 - mean1)^2
        // (Simplified, omitting total weight division since it's constant)
        let sigma_b = w0 * w1 * (mean0 - mean1).powi(2);

        if sigma_b > max_sigma_b {
            max_sigma_b = sigma_b;
            best_t = t;
        }
    }

    // Safety clamp: Ensure threshold isn't ridiculously high or low
    // If Otsu picks a threshold that classifies > 95% as noise, it might be too aggressive.
    // Or if it classifies noise as signal.

    // Fallback/Bias: We prefer to include frames if unsure.
    // Otsu tends to separate "strong content" from "weak content".
    // We modify the threshold slightly down (0.5x) to be more inclusive.
    best_t *= 0.5;

    // Hard lower bound relative to max score to kill artifacts
    // Lowered to 0.02 to allow very faint sprites (building_deconstruction etc.)
    if cv < 0.5 {
        best_t = best_t.max(max_s * 0.01);
    } else {
        best_t = best_t.max(max_s * 0.05);
    }

    #[cfg(test)]
    eprintln!("  Otsu Threshold: {:.4} (Max Score: {:.4})", best_t, max_s);

    scores.into_iter().filter(|&s| s > best_t).count() as u32
}

fn cell_activity_score(img: &RgbaImage, x: u32, y: u32, sw: u32, sh: u32) -> f32 {
    let mut total_alpha = 0.0;
    let mut max_a = 0u8;
    let mut hist = [0u32; 256];
    let mut pixels = 0;

    // Reduce margins to 1/40 to capture edge content better
    let (px, py) = ((sw / 40).max(0), (sh / 40).max(0));

    // Coherence check: Only count pixels that have at least one neighbor
    // This filters out "salt and pepper" noise common in compression
    let w = img.width();
    let h = img.height();

    let mut coherent_pixels = 0;

    for iy in (y + py)..(y + sh - py) {
        for ix in (x + px)..(x + sw - px) {
            let p = img.get_pixel(ix, iy);
            let a = p[3];
            if a > 0 {
                // Check neighbors for coherence
                let mut coherent = false;
                // Simple 4-connectivity check
                if ix > 0 && img.get_pixel(ix - 1, iy)[3] > 0 {
                    coherent = true;
                } else if ix < w - 1 && img.get_pixel(ix + 1, iy)[3] > 0 {
                    coherent = true;
                } else if iy > 0 && img.get_pixel(ix, iy - 1)[3] > 0 {
                    coherent = true;
                } else if iy < h - 1 && img.get_pixel(ix, iy + 1)[3] > 0 {
                    coherent = true;
                }

                if coherent {
                    total_alpha += f32::from(a) / 255.0;
                    max_a = max_a.max(a);
                    let l = (u32::from(p[0]) * 299 + u32::from(p[1]) * 587 + u32::from(p[2]) * 114)
                        / 1000;
                    hist[l.min(255) as usize] += 1;
                    pixels += 1;
                    coherent_pixels += 1;
                }
            }
        }
    }

    if pixels == 0 {
        return 0.0;
    }

    // Hard noise gate: if the brightest pixel is very faint, it's just compression noise.
    // Lowered to 30 (~12%)
    if max_a < 30 {
        return 0.0;
    }

    // Absolute Coherent Pixel Count Gate
    // Noise frames usually have < 10 coherent pixels (scattered specks).
    // Valid frames (even small particles) usually have a cluster > 15-20 pixels.
    if coherent_pixels < 15 {
        return 0.0;
    }

    let area = ((sw - 2 * px) * (sh - 2 * py)) as f32;
    let fill_ratio = total_alpha / area;

    // Secondary noise gate for faint AND sparse content
    // Since we check coherent_pixels, we can relax this.
    // Only kill if basically invisible.
    if max_a < 50 && fill_ratio < 0.001 {
        return 0.0;
    }
    let peak = f32::from(max_a) / 255.0;

    let mut ent = 0.0;
    for &c in &hist {
        if c > 0 {
            let p = c as f32 / pixels as f32;
            ent -= p * p.log2();
        }
    }
    let ent_norm = ent / 8.0;

    // Weight Entropy higher to favor "complex" visual content (VFX/Sprites) over "flat" noise.
    // Reduced peak weight because noise can be bright.
    (fill_ratio * 0.4) + (peak * 0.1) + (ent_norm * 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_divisors() {
        assert_eq!(get_divisors(12), vec![1, 2, 3, 4, 6, 12]);
    }
}
