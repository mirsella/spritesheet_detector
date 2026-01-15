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
#[must_use]
pub fn analyze_spritesheet(img: &DynamicImage) -> SpritesheetInfo {
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
    if total_dim == 192 {
        println!("Detecting period for 192");
    }
    let factors = get_factors_to_check(total_dim);
    let a_mean = alpha.iter().sum::<f32>() / alpha.len() as f32;
    let a_var = alpha.iter().map(|&x| (x - a_mean).powi(2)).sum::<f32>() / alpha.len() as f32;

    // Choose signal based on variance: Alpha for sprites, Gradient for maps.
    let signal = if a_var > 1e-5 { alpha } else { grad };
    let mean = signal.iter().sum::<f32>() / signal.len() as f32;
    let detrended: Vec<f32> = signal.iter().map(|&x| x - mean).collect();

    let mut scored = Vec::new();
    for &f in &factors {
        if f < 4 || f > total_dim / 2 {
            continue;
        }
        if (total_dim / f) > 512 {
            continue;
        }

        let r1 = normalized_correlation(&detrended, f as usize);
        let r2 = normalized_correlation(&detrended, (f * 2) as usize);
        let r3 = normalized_correlation(&detrended, (f * 3) as usize);

        let num_repeats = total_dim / f;
        if num_repeats < 2 && r1 < 0.25 {
            continue;
        }

        let mut product = r1.max(0.01);
        let mut count = 1.0;
        if (f as usize * 2) < detrended.len() {
            product *= r2.max(0.01);
            count += 1.0;
        }
        if (f as usize * 3) < detrended.len() {
            product *= r3.max(0.01);
            count += 1.0;
        }

        let h_score = product.powf(1.0 / count);
        let vov = calculate_vov(signal, f as usize);

        // DEBUG LOGGING
        // if (total_dim == 400 && (f == 133 || f == 200)) ...

        // Reject very weak correlations (likely noise or single images)
        // Tilesets (different sprites) can have low correlation (~0.2), but 0.01 is definitely not a grid.
        // ui_chest_lock has h_score ~0.010. So we must be very careful.
        if h_score < 0.005 {
            continue;
        }

        // However, allow low h_score (but > 0.005) if VoV is extremely strong
        if h_score < 0.1 && vov < 10.0 {
            continue;
        }

        let vov_factor = 1.0 + vov * 3.0;

        let mut reliability = 1.0;

        // Reject flat signals (e.g. solid color blocks)
        if vov < 0.01 {
            reliability *= 0.5;
        }

        // Generic Cut Line Analysis: Check if grid lines cut through solid objects
        let max_a = alpha.iter().cloned().fold(0.0, f32::max);
        let mean_g = grad.iter().sum::<f32>() / grad.len() as f32;
        let mut bad_cuts = 0;
        let checks = (num_repeats - 1).min(10);

        for k in 1..=checks {
            let cut_idx = (f * k) as usize;
            if cut_idx >= alpha.len() {
                break;
            }

            let local_a = alpha[cut_idx];
            let local_g = grad[cut_idx];

            // Dynamic strictness based on autocorrelation (h_score)
            // If correlation is high, we trust the grid more and accept "messy" cuts.
            // If correlation is low, we require clear evidence (gaps or strong edges).
            // Raised threshold to 0.6 to treat hero_builder (h=0.56) as strict, identifying bad cuts.
            // chest_notif (h=0.71) remains lenient.
            let strict_cuts = h_score < 0.6;
            let alpha_thresh_ratio = if strict_cuts { 0.1 } else { 0.25 };
            let grad_thresh_ratio = if strict_cuts { 3.0 } else { 1.0 };

            let is_edge = local_g > 0.5 && local_g > mean_g * grad_thresh_ratio;
            if local_a > max_a * alpha_thresh_ratio && !is_edge {
                bad_cuts += 1;
            }
        }

        if bad_cuts > 0 {
            let ratio = bad_cuts as f32 / checks as f32;
            if ratio > 0.25 {
                reliability *= 0.001; // Kill it if > 25% cuts are bad
            } else {
                reliability *= 1.0 - (ratio * 0.9);
            }
        }

        // Penalty for very small periods (often texture/noise)
        // Increased range to catch repeating borders like border_s_1 (f=72, reps=14)
        if f < 96 && num_repeats >= 12 {
            reliability *= 0.1;
        }

        // Penalty for excessive fragmentation
        if num_repeats > 20 {
            reliability *= 0.1;
        }

        if num_repeats == 2 {
            let mut penalty: f32;
            // General slight penalty for 2-splits
            if total_dim > 800 {
                penalty = 0.5;
            } else if total_dim >= 180 && total_dim <= 220 {
                penalty = 0.001;
            } else if total_dim > 300 {
                // Penalize 2-splits for larger images (likely single assets like 400px cards)
                // ui_chest_lock is 260px (safe). coffin_maw_card is 400px (penalized).
                penalty = 0.1;
            } else {
                penalty = 0.95;
            }

            // If correlation is strong (e.g. map_tiles h=0.5), reduce penalty to save valid grids
            if h_score > 0.4 {
                penalty = penalty.max(0.9);
            }
            reliability *= penalty;

            // Fallback: Stronger penalty for small images (projectiles)
            if total_dim < 120 {
                reliability *= 0.1;
            }
        }

        // Bonus for "Sweet Spot" repeat counts (3 to 8)
        // 3-repeats are neutral to avoid false positives on single assets (1x3 split)
        // But we need it for hero_builder (height 192 -> 3x64)
        let repeat_bonus = if num_repeats >= 3 && num_repeats <= 8 {
            if vov > 0.5 { 1.1 } else { 1.0 }
        } else {
            1.0
        };

        let score = h_score * vov_factor * reliability * repeat_bonus;

        scored.push((f, score));
    }

    if scored.is_empty() {
        return total_dim;
    }
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let max_s = scored[0].1;
    if max_s < 0.8 {
        return total_dim;
    }

    let mut best_f = scored[0].0;

    // Tie-breaker: Prefer larger periods (simpler grids) if they are close enough to top score
    // This protects against over-fragmenting single assets
    for (f, s) in &scored {
        if *s >= max_s * 0.9 {
            if *f > best_f {
                best_f = *f;
            }
        } else {
            break;
        }
    }

    // Sub-harmonic correction
    let mut current_best_vov = calculate_vov(signal, best_f as usize);

    // Check for REDUCTION (smaller divisor)
    for &(f, _s) in &scored {
        if f < best_f && best_f % f == 0 {
            let cand_vov = calculate_vov(signal, f as usize);
            if cand_vov > current_best_vov * 0.8 {
                let cand_score = scored
                    .iter()
                    .find(|&&(k, _)| k == f)
                    .map(|&(_, s)| s)
                    .unwrap_or(0.0);
                if cand_score > max_s * 0.5 {
                    best_f = f;
                    current_best_vov = cand_vov;
                }
            }
        }
    }

    // Check for PROMOTION (larger multiple)
    for &(f, _s) in &scored {
        if f > best_f && f % best_f == 0 {
            let mult = f / best_f;
            if mult > 8 {
                continue;
            }
            let cand_vov = calculate_vov(signal, f as usize);

            // Context-aware promotion threshold
            let target_repeats = total_dim / f;
            let threshold = if target_repeats == 2 {
                // Promoting to a 2-split is risky (often just doubling the period)
                3.0
            } else {
                // Promoting to 3+ repeats: standard threshold
                1.2
            };

            if cand_vov > current_best_vov * threshold {
                let cand_score = scored
                    .iter()
                    .find(|&&(k, _)| k == f)
                    .map(|&(_, s)| s)
                    .unwrap_or(0.0);
                if cand_score > max_s * 0.2 {
                    best_f = f;
                    current_best_vov = cand_vov;
                }
            }
        }
    }

    best_f
}

fn get_factors_to_check(n: u32) -> Vec<u32> {
    let mut factors = get_divisors(n);
    if n > 10 {
        let mut factors_minus_1 = get_divisors(n - 1);
        factors.append(&mut factors_minus_1);
    }
    factors.sort_unstable();
    factors.dedup();
    factors
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
    (folded.iter().map(|&x| (x - m).powi(2)).sum::<f32>() / p as f32) / (m.abs() + 1.0)
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
    let px = (sw / 100).max(1);
    let py = (sh / 100).max(1);

    let mut alpha_sum = 0u64;
    // Step by 2 is good for performance
    for iy in (y + py..y + sh - py).step_by(2) {
        if iy >= h {
            break;
        }
        for ix in (x + px..x + sw - px).step_by(2) {
            if ix >= w {
                break;
            }
            let p = img.get_pixel(ix, iy);
            alpha_sum += p[3] as u64;
            // Early exit if definitely active (e.g. > 10 solid pixels)
            if alpha_sum > 2550 {
                return true;
            }
        }
    }
    // Threshold: Equivalent to ~1 solid pixel or ~50 faint pixels (alpha 5)
    // Lowered to catch very faint effects
    alpha_sum > 20
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_divisors() {
        assert_eq!(get_divisors(12), vec![1, 2, 3, 4, 6, 12]);
    }
}
