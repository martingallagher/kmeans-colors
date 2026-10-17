use std::convert::TryFrom;

use palette::{FromColor, LinSrgb, Oklab, Srgb};

use crate::{Calculate, CentroidData};

#[derive(Clone, Default)]
struct Bin {
    count: u64,
    pixel: [u8; 3],
    center_distance: u16,
}

// Precomputed from palette 0.7.7's byte-to-f32-to-linear path to preserve its f32 rounding.
#[rustfmt::skip]
const BYTE_GAMMA: [f32; 256] = [
    0.0, 0.000303527, 0.000607054, 0.00091058103, 0.001214108, 0.001517635,
    0.0018211621, 0.002124689, 0.002428216, 0.002731743, 0.00303527, 0.003346535,
    0.0036765062, 0.0040247166, 0.004391441, 0.0047769523, 0.005181515, 0.0056053908,
    0.0060488316, 0.006512089, 0.0069954083, 0.0074990294, 0.008023192, 0.008568125,
    0.009134057, 0.009721216, 0.01032982, 0.010960094, 0.011612245, 0.012286487,
    0.012983031, 0.013702081, 0.014443841, 0.015208512, 0.01599629, 0.016807375,
    0.017641952, 0.018500218, 0.019382358, 0.020288559, 0.021219006, 0.02217388,
    0.023153365, 0.02415763, 0.025186857, 0.026241219, 0.027320888, 0.028426034,
    0.029556833, 0.03071344, 0.03189603, 0.033104766, 0.034339808, 0.035601314,
    0.036889445, 0.038204376, 0.039546236, 0.0409152, 0.04231141, 0.043735027,
    0.045186203, 0.046665084, 0.048171822, 0.049706563, 0.051269453, 0.05286064,
    0.05448027, 0.056128483, 0.057805423, 0.05951123, 0.06124604, 0.06301002,
    0.064803265, 0.06662594, 0.06847817, 0.070360094, 0.072271846, 0.074213564,
    0.076185375, 0.07818741, 0.08021981, 0.0822827, 0.0843762, 0.08650045,
    0.08865557, 0.090841696, 0.093058966, 0.09530746, 0.09758735, 0.099898726,
    0.102241725, 0.10461648, 0.1070231, 0.1094617, 0.11193242, 0.11443536,
    0.11697067, 0.119538434, 0.122138776, 0.12477182, 0.12743768, 0.13013647,
    0.13286832, 0.13563335, 0.13843162, 0.1412633, 0.14412849, 0.14702728,
    0.1499598, 0.15292616, 0.15592647, 0.15896083, 0.16202937, 0.1651322,
    0.1682694, 0.1714411, 0.17464739, 0.17788841, 0.18116423, 0.184475,
    0.18782079, 0.19120167, 0.19461781, 0.1980693, 0.20155624, 0.2050787,
    0.20863685, 0.21223079, 0.21586053, 0.21952623, 0.22322798, 0.22696589,
    0.23074007, 0.2345506, 0.23839758, 0.24228114, 0.24620134, 0.25015828,
    0.2541521, 0.25818285, 0.26225066, 0.2663556, 0.2704978, 0.2746773,
    0.27889425, 0.28314874, 0.28744084, 0.29177064, 0.29613826, 0.30054379,
    0.30498728, 0.3094689, 0.3139887, 0.31854674, 0.32314318, 0.32777807,
    0.3324515, 0.33716357, 0.3419144, 0.3467041, 0.35153264, 0.35640016,
    0.3613068, 0.3662526, 0.3712377, 0.37626213, 0.38132602, 0.38642943,
    0.39157248, 0.39675522, 0.40197778, 0.4072402, 0.4125426, 0.41788507,
    0.42326766, 0.42869046, 0.43415362, 0.43965715, 0.44520116, 0.45078576,
    0.45641097, 0.46207696, 0.46778375, 0.47353145, 0.47932014, 0.4851499,
    0.4910208, 0.49693292, 0.5028864, 0.5088813, 0.5149177, 0.5209956,
    0.5271152, 0.5332765, 0.5394796, 0.5457246, 0.5520115, 0.5583405,
    0.56471163, 0.5711249, 0.5775805, 0.5840785, 0.5906189, 0.5972019,
    0.6038274, 0.6104956, 0.61720663, 0.62396044, 0.6307572, 0.63759696,
    0.64447975, 0.6514057, 0.65837485, 0.66538733, 0.6724432, 0.67954254,
    0.6866853, 0.6938718, 0.7011019, 0.7083758, 0.71569353, 0.7230551,
    0.73046076, 0.73791045, 0.74540436, 0.7529423, 0.76052463, 0.7681513,
    0.77582234, 0.7835379, 0.79129803, 0.79910284, 0.80695236, 0.8148467,
    0.82278585, 0.83076996, 0.8387991, 0.8468733, 0.8549927, 0.8631573,
    0.8713672, 0.87962246, 0.8879232, 0.8962694, 0.90466124, 0.9130987,
    0.9215819, 0.9301109, 0.9386858, 0.9473066, 0.9559734, 0.9646863,
    0.9734453, 0.9822506, 0.9911021, 1.0,
];

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
    let gamma = &BYTE_GAMMA;
    let colors: Vec<Oklab> = bins
        .iter()
        .map(|bin| {
            Oklab::from_color(LinSrgb::new(
                gamma[usize::from(bin.pixel[0])],
                gamma[usize::from(bin.pixel[1])],
                gamma[usize::from(bin.pixel[2])],
            ))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_gamma_matches_float_conversion_exactly() {
        for channel in 0..=255_u8 {
            let previous: LinSrgb<f32> = Srgb::new(channel, channel, channel)
                .into_format::<f32>()
                .into_linear();

            assert_eq!(
                BYTE_GAMMA[usize::from(channel)].to_bits(),
                previous.red.to_bits()
            );
        }
    }

    #[test]
    fn single_color_palettes_preserve_float_centroids_for_all_channels() {
        for red in 0..=255_u8 {
            let pixel = Srgb::new(red, red.wrapping_add(85), red.wrapping_add(170));
            let expected: Oklab = Oklab::from_color(pixel.into_format::<f32>().into_linear());
            let palette = get_diverse_palette(8, [pixel; 3]);

            assert_eq!(palette.len(), 1);
            assert_eq!(palette[0].centroid.l.to_bits(), expected.l.to_bits());
            assert_eq!(palette[0].centroid.a.to_bits(), expected.a.to_bits());
            assert_eq!(palette[0].centroid.b.to_bits(), expected.b.to_bits());
            assert_eq!(palette[0].percentage, 1.0);
            assert_eq!(palette[0].index, 0);
        }
    }
}
