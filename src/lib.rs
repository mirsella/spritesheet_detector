use image::{DynamicImage, RgbaImage};
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
    let rgba = img.to_rgba8();
    let (width, height) = rgba.dimensions();
    let (v_alpha, h_alpha, v_grad, h_grad) = extract_signals(&rgba);
    let widths = score_periods(&v_alpha, &v_grad, width);
    let heights = score_periods(&h_alpha, &h_grad, height);
    let initial_width = select_period(&widths, width, false);
    let initial_height = select_period(&heights, height, false);

    // Two dissimilar rows or columns can still be animation frames when the
    // perpendicular axis independently establishes a grid. Without that
    // evidence, anticorrelated halves of an outline are an ambiguous split.
    let sprite_width = if initial_height < height {
        select_period(&widths, width, true)
    } else {
        initial_width
    };
    let sprite_height = if initial_width < width {
        select_period(&heights, height, true)
    } else {
        initial_height
    };

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

    let frame_count = if sprite_width > 0 && sprite_height > 0 {
        find_last_active_frame(&rgba, sprite_width, sprite_height, columns, rows)
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

fn extract_signals(rgba: &RgbaImage) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let (w, h) = rgba.dimensions();

    // Premultiplied luminance keeps invisible RGB out of the gradient signal.
    let gray: Vec<u8> = rgba
        .pixels()
        .map(|p| {
            ((u32::from(p[0]) * 299 + u32::from(p[1]) * 587 + u32::from(p[2]) * 114)
                * u32::from(p[3])
                / (1000 * 255)) as u8
        })
        .collect();

    let (v_alpha, v_grad) = (0..w)
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
        .unzip();

    let (h_alpha, h_grad) = (0..h)
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
        .unzip();

    (v_alpha, h_alpha, v_grad, h_grad)
}

struct PeriodCandidate {
    period: u32,
    remainder: u32,
    score: f32,
    variance: f32,
    requires_perpendicular_grid: bool,
}

fn score_periods(alpha: &[f32], grad: &[f32], total_dim: u32) -> Vec<PeriodCandidate> {
    let a_mean = alpha.iter().sum::<f32>() / alpha.len() as f32;
    let a_var = alpha.iter().map(|&x| (x - a_mean).powi(2)).sum::<f32>() / alpha.len() as f32;
    let max_a = alpha.iter().copied().fold(0.0, f32::max);
    let mean_g = grad.iter().sum::<f32>() / grad.len() as f32;

    // Choose signal based on variance: Alpha for sprites, Gradient for maps.
    let signal = if a_var > 1e-5 { alpha } else { grad };
    let mean = signal.iter().sum::<f32>() / signal.len() as f32;
    let detrended: Vec<f32> = signal.iter().map(|&x| x - mean).collect();

    let mut scored = Vec::new();
    for (f, rem) in get_factors_to_check(total_dim) {
        if f < 4 || f > total_dim / 2 {
            continue;
        }
        if (total_dim / f) > 512 {
            continue;
        }

        let r1 = normalized_correlation(&detrended, f as usize);
        let num_repeats = total_dim / f;
        if r1 < 0.01 && num_repeats > 2 {
            continue;
        }
        let r2 = normalized_correlation(&detrended, (f * 2) as usize);
        let r3 = normalized_correlation(&detrended, (f * 3) as usize);

        let mut product = r1.max(0.01);
        let mut count = 1.0;
        // Ignore checks with very small overlap (< 8 pixels) to avoid noise
        if (f as usize * 2) < detrended.len().saturating_sub(8) {
            product *= r2.max(0.01);
            count += 1.0;
        }
        if (f as usize * 3) < detrended.len().saturating_sub(8) {
            product *= r3.max(0.01);
            count += 1.0;
        }

        let h_score = product.powf(1.0 / count);
        let vov = calculate_vov(signal, f as usize);

        // Weak correlations need strong folded variance to remain plausible.
        if h_score < 0.1 && vov < 10.0 {
            continue;
        }

        let vov_factor = 1.0 + vov * 3.0;

        // Penalty for non-exact total dimension (from resizing/padding)
        let remainder_penalty = 1.0 / (1.0 + rem as f32 * 0.2);

        let mut reliability = 1.0;

        // Reject flat signals (e.g. solid color blocks)
        if vov < 0.01 {
            reliability *= 0.5;
        }

        // Generic Cut Line Analysis: Check if grid lines cut through solid objects
        let mut bad_cuts = 0;
        let checks = (num_repeats - 1).min(10);

        // Approximate periods need clearer gaps: an internal edge can
        // otherwise turn a padded outline into a plausible resized grid.
        let strict_cuts = h_score < 0.6 || rem > 0;
        let alpha_thresh_ratio = if strict_cuts && (rem > 0 || r1 < 0.4) {
            0.1
        } else {
            0.25
        };
        let grad_thresh_ratio = if strict_cuts { 3.0 } else { 1.0 };

        for k in 1..=checks {
            let cut_idx = (f * k) as usize;
            if cut_idx >= alpha.len() {
                break;
            }

            let local_a = alpha[cut_idx];
            let local_g = grad[cut_idx];

            // With varying alpha, a color edge alone does not establish a
            // frame boundary for a weak or approximate candidate. Uniform-
            // alpha sheets still need gradients because they have no gaps.
            let is_edge = (!strict_cuts || a_var <= 1e-5)
                && local_g > 0.5
                && local_g > mean_g * grad_thresh_ratio;
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
        // We only apply this if the variance is low, to avoid killing valid small-period spritesheets
        if f < 128 && num_repeats >= 12 && vov < 100.0 {
            reliability *= 0.1;
        }

        // Texture/Border penalty: High repeats with weak correlation
        if num_repeats >= 8 && h_score < 0.40 {
            reliability *= 0.1;
        }

        // Penalty for excessive fragmentation
        if num_repeats > 20 {
            reliability *= 0.1;
        }

        if num_repeats == 2 {
            let mut penalty: f32;

            // KILL APPROXIMATE 2-SPLITS
            if rem > 0 && h_score < 0.95 {
                reliability = 0.0;
            }

            // Special case for "Sweet Spot" 2-row sprites
            if total_dim >= 180 && total_dim <= 220 {
                penalty = 0.001;
            } else if total_dim > 300 {
                penalty = 0.1;
            } else {
                penalty = 0.95;
            }

            if h_score > 0.4 && !(total_dim >= 180 && total_dim <= 220) {
                penalty = penalty.max(0.9);
            } else if total_dim >= 180 && total_dim <= 220 && h_score > 0.95 {
                penalty = penalty.max(0.9);
            }
            reliability *= penalty;

            // Fallback: Stronger penalty for small images (projectiles)
            if total_dim < 100 {
                reliability *= 0.01;
            }
        }

        // Bonus for "Sweet Spot" repeat counts (3 to 8)
        let repeat_bonus = if num_repeats >= 3 && num_repeats <= 8 {
            // Evolving frames may correlate with their neighbor but not with
            // distant frames. Do not classify that as weak adjacent repetition.
            if r1 < 0.25 && ((f < 64 && vov < 10.0) || vov < 5.0) {
                reliability *= 0.1;
                1.0
            } else if vov > 0.5 {
                1.2
            } else {
                1.0
            }
        } else {
            1.0
        };

        let score = h_score * vov_factor * reliability * repeat_bonus * remainder_penalty;
        scored.push(PeriodCandidate {
            period: f,
            remainder: rem,
            score,
            variance: vov,
            requires_perpendicular_grid: r1 < 0.01,
        });
    }
    scored.sort_by(|a, b| b.score.total_cmp(&a.score));
    scored
}

fn select_period(scored: &[PeriodCandidate], total_dim: u32, has_perpendicular_grid: bool) -> u32 {
    let candidates = scored
        .iter()
        .filter(|candidate| has_perpendicular_grid || !candidate.requires_perpendicular_grid);
    let Some(strongest) = candidates.clone().next() else {
        return total_dim;
    };
    let max_s = strongest.score;
    if max_s < 0.8 {
        return total_dim;
    }

    let mut best = strongest;

    for candidate in candidates.clone() {
        if candidate.score >= max_s * 0.96 {
            if candidate.remainder > best.remainder {
                continue;
            }
            if candidate.period > best.period {
                best = candidate;
            }
        } else {
            break;
        }
    }

    // Check for REDUCTION (smaller divisor)
    for candidate in candidates.clone() {
        if candidate.period < best.period {
            let ratio = best.period as f32 / candidate.period as f32;
            let nearest_mult = ratio.round();

            if nearest_mult < 2.0 {
                continue;
            }

            if (ratio - nearest_mult).abs() < 0.01
                && candidate.variance > best.variance * 0.6
                && candidate.score > max_s * 0.4
            {
                best = candidate;
            }
        }
    }

    // Check for PROMOTION (larger multiple)
    for candidate in candidates {
        if candidate.period > best.period && candidate.period % best.period == 0 {
            let mult = candidate.period / best.period;
            if mult > 8 {
                continue;
            }
            let target_repeats = total_dim / candidate.period;
            let threshold = if target_repeats == 2 { 3.0 } else { 1.2 };

            if candidate.variance > best.variance * threshold && candidate.score > max_s * 0.2 {
                best = candidate;
            }
        }
    }

    best.period
}

fn get_factors_to_check(n: u32) -> Vec<(u32, u32)> {
    let mut factors = Vec::new();
    let range = if n > 16 { 16 } else { 0 };

    for i in 0..=range {
        let f_list = get_divisors(n - i);
        for f in f_list {
            factors.push((f, i));
        }
    }

    factors.sort_unstable_by(|a, b| match a.0.cmp(&b.0) {
        std::cmp::Ordering::Equal => a.1.cmp(&b.1),
        other => other,
    });
    factors.dedup_by(|a, b| a.0 == b.0);
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

fn find_last_active_frame(rgba: &RgbaImage, sw: u32, sh: u32, cols: u32, rows: u32) -> u32 {
    let total_frames = rows * cols;
    let mut last_active = 0;

    for i in 0..total_frames {
        let r = i / cols;
        let c = i % cols;
        let x = c * sw;
        let y = r * sh;

        if is_frame_active(rgba, x, y, sw, sh) {
            last_active = i + 1;
        }
    }
    last_active
}

fn is_frame_active(img: &RgbaImage, x: u32, y: u32, sw: u32, sh: u32) -> bool {
    let (w, h) = img.dimensions();
    let px = (sw / 100).max(1);
    let py = (sh / 100).max(1);

    let mut alpha_sum = 0u64;
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
            if alpha_sum > 2550 {
                return true;
            }
        }
    }
    alpha_sum > 20
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ambiguous_candidates_do_not_change_the_independent_grid_score() {
        // An ambiguous candidate must neither change the independent maximum
        // nor reenter the result through harmonic promotion.
        for (score, variance) in [(100.0, 2.0), (9.0, 4.0)] {
            let mut candidates = [
                PeriodCandidate {
                    period: 64,
                    remainder: 0,
                    score,
                    variance,
                    requires_perpendicular_grid: true,
                },
                PeriodCandidate {
                    period: 32,
                    remainder: 0,
                    score: 10.0,
                    variance: 1.0,
                    requires_perpendicular_grid: false,
                },
            ];
            candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
            assert_eq!(select_period(&candidates, 128, false), 32);
            assert_eq!(select_period(&candidates, 128, true), 64);
        }
    }

    #[test]
    fn test_divisors() {
        assert_eq!(get_divisors(12), vec![1, 2, 3, 4, 6, 12]);
    }
}
