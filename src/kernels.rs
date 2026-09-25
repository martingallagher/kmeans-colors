use crate::Calculate;
#[cfg(feature = "palette_color")]
use crate::{Hamerly, HamerlyCentroids, HamerlyPoint};

pub(crate) fn update_distances<C: Calculate>(buf: &[C], center: &C, distances: &mut [f32]) -> f32 {
    #[cfg(feature = "simd")]
    {
        fearless_simd::dispatch!(fearless_simd::Level::new(), s => simd::update_distances(s, buf, center, distances))
    }
    #[cfg(not(feature = "simd"))]
    update_distances_scalar(buf, center, distances, 0.0)
}

fn update_distances_scalar<C: Calculate>(
    buf: &[C],
    center: &C,
    distances: &mut [f32],
    mut sum: f32,
) -> f32 {
    for (point, nearest) in buf.iter().zip(distances) {
        let distance = C::difference(point, center);
        if distance < *nearest {
            *nearest = distance;
        }
        sum += *nearest;
    }
    sum
}

#[cfg(feature = "palette_color")]
pub(crate) fn assign<C: Calculate>(buf: &[C], centers: &[C], indices: &mut Vec<u8>) {
    #[cfg(feature = "simd")]
    fearless_simd::dispatch!(fearless_simd::Level::new(), s => simd::assign(s, buf, centers, indices));
    #[cfg(not(feature = "simd"))]
    assign_scalar(buf, centers, indices);
}

#[cfg(feature = "palette_color")]
#[allow(clippy::cast_possible_truncation)]
fn assign_scalar<C: Calculate>(buf: &[C], centers: &[C], indices: &mut Vec<u8>) {
    for point in buf {
        let (mut nearest, mut index) = (f32::MAX, 0);
        for (i, center) in centers.iter().enumerate() {
            let distance = C::difference(point, center);
            if distance < nearest {
                nearest = distance;
                index = i;
            }
        }
        indices.push(index as u8);
    }
}

#[cfg(feature = "palette_color")]
pub(crate) fn assign_hamerly<C: Hamerly>(
    buf: &[C],
    centers: &HamerlyCentroids<C>,
    points: &mut [HamerlyPoint],
) {
    #[cfg(feature = "simd")]
    fearless_simd::dispatch!(fearless_simd::Level::new(), s => simd::assign_hamerly(s, buf, centers, points));
    #[cfg(not(feature = "simd"))]
    assign_hamerly_scalar(buf, centers, points);
}

#[cfg(all(feature = "palette_color", any(not(feature = "simd"), test)))]
#[allow(clippy::cast_possible_truncation)]
fn assign_hamerly_scalar<C: Hamerly>(
    buf: &[C],
    centers: &HamerlyCentroids<C>,
    points: &mut [HamerlyPoint],
) {
    for (val, point) in buf.iter().zip(points) {
        let z = centers.half_distances[point.index as usize].max(point.lower_bound);
        if point.upper_bound <= z {
            continue;
        }
        point.upper_bound = C::difference(val, &centers.centroids[point.index as usize]).sqrt();
        if point.upper_bound <= z || centers.centroids.len() < 2 {
            continue;
        }
        let mut min1 = C::difference(val, &centers.centroids[0]);
        let (mut min2, mut c1) = (f32::MAX, 0);
        for (j, center) in centers.centroids.iter().enumerate().skip(1) {
            let diff = C::difference(val, center);
            if diff < min1 {
                min2 = min1;
                min1 = diff;
                c1 = j;
            } else if diff < min2 {
                min2 = diff;
            }
        }
        if c1 as u8 != point.index {
            point.index = c1 as u8;
            point.upper_bound = min1.sqrt();
        }
        point.lower_bound = min2.sqrt();
    }
}

#[cfg(feature = "simd")]
mod simd {
    use super::*;
    use fearless_simd::{f32x8, prelude::*};

    // Keep each lane's scalar distance arithmetic intact, including f64 inputs.
    // Inlining lets LLVM vectorize independent calls without changing the public traits.
    #[inline(always)]
    pub(super) fn update_distances<S: Simd, C: Calculate>(
        s: S,
        buf: &[C],
        center: &C,
        distances: &mut [f32],
    ) -> f32 {
        let end = buf.len().min(distances.len()) / 8 * 8;
        let mut sum = 0.0;
        for (pixels, nearest) in buf[..end]
            .as_chunks::<8>()
            .0
            .iter()
            .zip(distances[..end].as_chunks_mut::<8>().0.iter_mut())
        {
            let distance = f32x8::from_fn(s, |i| C::difference(&pixels[i], center));
            let old = f32x8::from_slice(s, nearest);
            distance
                .simd_lt(old)
                .select(distance, old)
                .store_slice(nearest);
            // Preserve addition order so seeded initialization remains reproducible.
            for &distance in nearest.iter() {
                sum += distance;
            }
        }
        update_distances_scalar(&buf[end..], center, &mut distances[end..], sum)
    }

    #[cfg(feature = "palette_color")]
    #[inline(always)]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub(super) fn assign<S: Simd, C: Calculate>(
        s: S,
        buf: &[C],
        centers: &[C],
        indices: &mut Vec<u8>,
    ) {
        let (chunks, tail) = buf.as_chunks::<8>();
        for pixels in chunks {
            let mut nearest = f32x8::splat(s, f32::MAX);
            let mut index = f32x8::splat(s, 0.0);
            for (i, center) in centers.iter().enumerate() {
                let distance = f32x8::from_fn(s, |lane| C::difference(&pixels[lane], center));
                let closer = distance.simd_lt(nearest);
                nearest = closer.select(distance, nearest);
                index = closer.select(f32x8::splat(s, f32::from(i as u8)), index);
            }
            indices.extend(index.as_array().iter().map(|&i| i as u8));
        }
        assign_scalar(tail, centers, indices);
    }

    #[cfg(feature = "palette_color")]
    #[inline(always)]
    pub(super) fn assign_hamerly<S: Simd, C: Hamerly>(
        s: S,
        buf: &[C],
        centers: &HamerlyCentroids<C>,
        points: &mut [HamerlyPoint],
    ) {
        // Only batch points which survive both bounds checks. Converged points
        // keep their cheap scalar skip, even when neighboring pixels need a search.
        let mut active = [0; 8];
        let mut count = 0;
        for i in 0..buf.len().min(points.len()) {
            let point = &mut points[i];
            let z = centers.half_distances[point.index as usize].max(point.lower_bound);
            if point.upper_bound <= z {
                continue;
            }
            point.upper_bound =
                C::difference(&buf[i], &centers.centroids[point.index as usize]).sqrt();
            if point.upper_bound <= z || centers.centroids.len() < 2 {
                continue;
            }
            active[count] = i;
            count += 1;
            if count == 8 {
                search_hamerly(s, buf, centers, points, &active);
                count = 0;
            }
        }
        if count != 0 {
            search_hamerly(s, buf, centers, points, &active[..count]);
        }
    }

    #[cfg(feature = "palette_color")]
    #[inline(always)]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn search_hamerly<S: Simd, C: Hamerly>(
        s: S,
        buf: &[C],
        centers: &HamerlyCentroids<C>,
        points: &mut [HamerlyPoint],
        active: &[usize],
    ) {
        // Repeat a valid point in unused tail lanes; only active lanes are stored.
        let pixels: [&C; 8] = core::array::from_fn(|i| &buf[*active.get(i).unwrap_or(&active[0])]);
        let mut min1 = f32x8::from_fn(s, |i| C::difference(pixels[i], &centers.centroids[0]));
        let mut min2 = f32x8::splat(s, f32::MAX);
        let mut index = f32x8::splat(s, 0.0);
        for (i, center) in centers.centroids.iter().enumerate().skip(1) {
            let distance = f32x8::from_fn(s, |lane| C::difference(pixels[lane], center));
            let closer = distance.simd_lt(min1);
            min2 = closer.select(min1, distance.simd_lt(min2).select(distance, min2));
            min1 = closer.select(distance, min1);
            index = closer.select(f32x8::splat(s, f32::from(i as u8)), index);
        }
        for (lane, &i) in active.iter().enumerate() {
            let point = &mut points[i];
            if index[lane] as u8 != point.index {
                point.index = index[lane] as u8;
                point.upper_bound = min1[lane].sqrt();
            }
            point.lower_bound = min2[lane].sqrt();
        }
    }
}

#[cfg(all(test, feature = "palette_color", feature = "simd"))]
mod tests {
    use super::*;
    use core::convert::TryFrom;
    use fearless_simd::{dispatch, Level};
    use palette::{Lab, Srgb};

    fn equal_float(actual: f32, expected: f32) {
        assert!(actual.to_bits() == expected.to_bits() || (actual.is_nan() && expected.is_nan()));
    }

    fn check<C: Hamerly + Clone>(buf: &[C]) {
        let levels = vec![Level::baseline(), Level::new()];
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        let levels = {
            let mut levels = levels;
            if let Some(s) = Level::new().as_sse4_2() {
                levels.push(Level::Sse4_2(s));
            }
            if let Some(s) = Level::new().as_avx2() {
                levels.push(Level::Avx2(s));
            }
            levels
        };
        for level in levels {
            for len in (0..18).chain([buf.len()]) {
                let pixels = &buf[..len];
                for k in [0, 1, 2, 8, 9, 20, 256] {
                    // Repeated centers exercise first-index tie handling.
                    let mut centers = HamerlyCentroids::new(k);
                    centers.centroids = buf.iter().cloned().cycle().take(k).collect();
                    if k > 1 {
                        centers.centroids[1] = centers.centroids[0].clone();
                    }
                    let mut expected = vec![255];
                    let mut actual = expected.clone();
                    assign_scalar(pixels, &centers.centroids, &mut expected);
                    dispatch!(level, s => simd::assign(s, pixels, &centers.centroids, &mut actual));
                    assert_eq!(actual, expected, "{level:?}, len={len}, k={k}");
                    if k == 0 {
                        continue;
                    }
                    C::compute_half_distances(&mut centers);
                    let mut reference: Vec<_> = (0..len)
                        .map(|i| HamerlyPoint {
                            index: u8::try_from(i * 7 % k).unwrap(),
                            upper_bound: if i % 5 == 0 { 0.0 } else { f32::MAX },
                            lower_bound: if i % 5 == 1 {
                                C::difference(&pixels[i], &centers.centroids[i * 7 % k]).sqrt()
                            } else {
                                0.0
                            },
                        })
                        .collect();
                    let mut actual = reference.clone();
                    assign_hamerly_scalar(pixels, &centers, &mut reference);
                    dispatch!(level, s => simd::assign_hamerly(s, pixels, &centers, &mut actual));
                    for (actual, reference) in actual.iter().zip(reference) {
                        assert_eq!(actual.index, reference.index);
                        equal_float(actual.upper_bound, reference.upper_bound);
                        equal_float(actual.lower_bound, reference.lower_bound);
                    }
                }
                let mut reference = vec![f32::MAX; len];
                let mut actual = reference.clone();
                for center in buf.iter().take(8) {
                    let expected_sum = update_distances_scalar(pixels, center, &mut reference, 0.0);
                    let sum = dispatch!(level, s => simd::update_distances(s, pixels, center, &mut actual));
                    equal_float(sum, expected_sum);
                    for (&actual, &expected) in actual.iter().zip(&reference) {
                        equal_float(actual, expected);
                    }
                }
            }
        }
    }

    #[test]
    #[allow(clippy::cast_possible_truncation)]
    fn simd_matches_scalar_for_lab_rgb_and_f64() {
        let components: Vec<_> = (0..257)
            .map(|i| {
                [
                    f64::from(i * 17 % 101),
                    f64::from(i * 37 % 255) - 128.0,
                    f64::from(i * 53 % 255) - 128.0,
                ]
            })
            .collect();
        let lab: Vec<Lab> = components
            .iter()
            .map(|&[l, a, b]| Lab::new(l as f32, a as f32, b as f32))
            .collect();
        let rgb: Vec<Srgb> = components
            .iter()
            .map(|&[r, g, b]| Srgb::new(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0))
            .collect();
        let lab64: Vec<Lab<palette::white_point::D65, f64>> = components
            .iter()
            .map(|&[l, a, b]| Lab::new(l + 1e10, a, b))
            .collect();
        let rgb64: Vec<Srgb<f64>> = components
            .iter()
            .map(|&[r, g, b]| Srgb::new(r / 255.0, g / 255.0, b / 255.0))
            .collect();
        check(&lab);
        check(&rgb);
        check(&lab64);
        check(&rgb64);
    }

    #[test]
    fn simd_preserves_nonfinite_comparisons() {
        let values = [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -0.0,
            0.0,
            f32::MAX,
            1.0,
            -1.0,
        ];
        let lab: Vec<Lab> = values
            .iter()
            .cycle()
            .take(23)
            .map(|&l| Lab::new(l, 0.0, 0.0))
            .collect();
        check(&lab);
    }
}
