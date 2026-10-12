use std::convert::TryFrom;

use palette::{FromColor, Oklab, Srgb};

use crate::{Calculate, CentroidData};

#[derive(Clone, Default)]
struct Bin {
    count: u64,
    pixel: [u8; 3],
    center_distance: u16,
}

/// Extract a palette that balances frequency and perceptual diversity.
///
/// All input pixels contribute to a 5-bit-per-channel sRGB histogram (at most
/// 32,768 bins). Each bin is represented by the observed pixel nearest its RGB
/// bin center, with lexicographic RGB tie breaking. Selection starts with the
/// most populated bin, then maximizes `sqrt(count) * distance_squared`, where
/// distance is measured in Oklab to the nearest previously selected color.
///
/// Returns at most `k` observed colors, in selection order, with matching indices.
/// Percentages are fractions of the original pixel count assigned to the nearest
/// selected color using each bin's representative, not the softened weights.
/// This histogram approximation may differ from assigning individual pixels.
/// Results are independent of input order and do not use a random seed.
///
/// Empty input or `k == 0` returns an empty palette. Handle transparency before
/// passing pixels in. Distinct colors in the same histogram bin are merged.
/// This favors small contrasting accents but cannot distinguish them from noise
/// or identify objects. It does not minimize image reconstruction error.
///
/// ```
/// use kmeans_colors::get_diverse_palette;
/// use palette::Srgb;
///
/// let pixels = std::iter::repeat_n(Srgb::new(128, 128, 128), 999)
///     .chain(std::iter::once(Srgb::new(255, 0, 0)));
/// let colors = get_diverse_palette(2, pixels);
/// assert_eq!(colors.len(), 2);
/// assert_eq!(colors[1].percentage, 0.001);
/// ```
///
/// # Panics
///
/// Panics if `k > 256`, the limit of [`CentroidData::index`].
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
pub fn get_diverse_palette(
    k: usize,
    pixels: impl IntoIterator<Item = Srgb<u8>>,
) -> Vec<CentroidData<Oklab>> {
    assert!(k <= 256, "palette size must be at most 256");

    if k == 0 {
        return Vec::new();
    }

    let mut bins = vec![Bin::default(); 32 * 32 * 32];

    for pixel in pixels {
        let rgb = [pixel.red, pixel.green, pixel.blue];
        let index = (usize::from(rgb[0] >> 3) << 10)
            | (usize::from(rgb[1] >> 3) << 5)
            | usize::from(rgb[2] >> 3);
        let bin = &mut bins[index];
        let distance = rgb
            .iter()
            .map(|&channel| (2 * i16::from(channel & 7) - 7).unsigned_abs().pow(2))
            .sum();

        if bin.count == 0 || (distance, rgb) < (bin.center_distance, bin.pixel) {
            bin.pixel = rgb;
            bin.center_distance = distance;
        }

        bin.count += 1;
    }

    let bins: Vec<_> = bins.into_iter().filter(|bin| bin.count != 0).collect();
    let colors: Vec<Oklab> = bins
        .iter()
        .map(|bin| {
            Oklab::from_color(
                Srgb::new(bin.pixel[0], bin.pixel[1], bin.pixel[2])
                    .into_format::<f32>()
                    .into_linear(),
            )
        })
        .collect();
    let weights: Vec<_> = bins.iter().map(|bin| (bin.count as f64).sqrt()).collect();
    let mut distances = vec![f32::INFINITY; bins.len()];
    let mut assignments = vec![0; bins.len()];
    let mut result = Vec::with_capacity(k.min(bins.len()));

    for index in 0..k.min(bins.len()) {
        let mut best = None;
        let mut best_score = 0.0;

        for (candidate, &weight) in weights.iter().enumerate() {
            let score = if index == 0 {
                weight
            } else {
                weight * f64::from(distances[candidate])
            };

            if score > best_score {
                best = Some(candidate);
                best_score = score;
            }
        }

        let Some(best) = best else {
            break;
        };

        result.push(CentroidData {
            centroid: colors[best],
            percentage: 0.0,
            index: u8::try_from(index).unwrap(),
        });

        for (candidate, color) in colors.iter().enumerate() {
            let distance = Oklab::difference(color, &colors[best]);

            if distance < distances[candidate] {
                distances[candidate] = distance;
                assignments[candidate] = index;
            }
        }
    }

    let mut counts = vec![0_u64; result.len()];
    let total: u64 = bins.iter().map(|bin| bin.count).sum();

    for (bin, index) in bins.iter().zip(assignments) {
        counts[index] += bin.count;
    }

    for (color, count) in result.iter_mut().zip(counts) {
        color.percentage = (count as f64 / total as f64) as f32;
    }

    result
}
