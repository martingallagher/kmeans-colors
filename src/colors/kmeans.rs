#[cfg(feature = "palette_color")]
use num_traits::{Float, FromPrimitive, Zero};
#[cfg(feature = "palette_color")]
use palette::{rgb::Rgb, rgb::Rgba, Lab};

use rand::{Rng, RngExt};

use crate::kmeans::{Calculate, Hamerly, HamerlyCentroids, HamerlyPoint};

#[cfg(feature = "palette_color")]
fn sum_clusters<C: Copy + Default + core::ops::AddAssign>(
    buf: &[C],
    indices: impl Iterator<Item = u8>,
    count: usize,
) -> Vec<(C, u64)> {
    let mut sums = vec![(C::default(), 0); count];
    for (index, &color) in indices.zip(buf) {
        if let Some((sum, count)) = sums.get_mut(index as usize) {
            *sum += color;
            *count += 1;
        }
    }
    sums
}

#[cfg(feature = "palette_color")]
fn compute_half_distances<C: Hamerly>(centers: &mut HamerlyCentroids<C>) {
    centers.half_distances.fill(f32::MAX);
    for (i, ci) in centers.centroids.iter().enumerate() {
        for (j, cj) in centers.centroids.iter().enumerate().skip(i + 1) {
            let distance = C::difference(ci, cj);
            centers.half_distances[i] = centers.half_distances[i].min(distance);
            centers.half_distances[j] = centers.half_distances[j].min(distance);
        }
    }
    for distance in &mut centers.half_distances {
        *distance = distance.sqrt() * 0.5;
    }
}

#[cfg(feature = "palette_color")]
fn update_bounds<C: Hamerly>(centers: &HamerlyCentroids<C>, points: &mut [HamerlyPoint]) {
    let (mut largest, mut second_largest, mut largest_index) = (0.0, 0.0, 0);
    for (index, &delta) in centers.deltas.iter().enumerate() {
        if delta > largest {
            second_largest = largest;
            largest = delta;
            largest_index = index;
        } else if delta > second_largest {
            second_largest = delta;
        }
    }

    for point in points {
        point.upper_bound += centers.deltas[point.index as usize];
        // The lower bound concerns every center except the assigned one.
        point.lower_bound -= if point.index as usize == largest_index {
            second_largest
        } else {
            largest
        };
    }
}

#[cfg(feature = "palette_color")]
impl<Wp, T> Calculate for Lab<Wp, T>
where
    T: Float + FromPrimitive + Zero,
    Lab<Wp, T>: core::ops::AddAssign<Lab<Wp, T>> + Default,
{
    fn get_closest_centroid(lab: &[Lab<Wp, T>], centroids: &[Lab<Wp, T>], indices: &mut Vec<u8>) {
        crate::kernels::assign(lab, centroids, indices);
    }

    #[allow(clippy::cast_precision_loss)]
    fn recalculate_centroids(
        mut rng: &mut impl Rng,
        buf: &[Lab<Wp, T>],
        centroids: &mut [Lab<Wp, T>],
        indices: &[u8],
    ) {
        let sums = sum_clusters(buf, indices.iter().copied(), centroids.len());
        for (cent, (temp, counter)) in centroids.iter_mut().zip(sums) {
            if counter != 0 {
                *cent = temp / T::from_f64(counter as f64).unwrap();
            } else {
                *cent = Self::create_random(&mut rng);
            }
        }
    }

    fn check_loop(centroids: &[Lab<Wp, T>], old_centroids: &[Lab<Wp, T>]) -> f32 {
        centroids
            .iter()
            .zip(old_centroids)
            .map(|(c0, c1)| Self::difference(c0, c1))
            .sum()
    }

    #[inline]
    fn create_random(rng: &mut impl Rng) -> Lab<Wp, T> {
        Lab::<Wp, T>::new(
            T::from_f64(rng.random_range(0.0..=100.0)).unwrap(),
            T::from_f64(rng.random_range(-128.0..=127.0)).unwrap(),
            T::from_f64(rng.random_range(-128.0..=127.0)).unwrap(),
        )
    }

    #[inline]
    fn difference(c1: &Lab<Wp, T>, c2: &Lab<Wp, T>) -> f32 {
        let temp = *c1 - *c2;

        ((temp.l).powi(2) + (temp.a).powi(2) + (temp.b).powi(2))
            .to_f32()
            .unwrap_or(f32::MAX)
    }
}

#[cfg(feature = "palette_color")]
impl<S, T> Calculate for Rgb<S, T>
where
    T: Float + FromPrimitive + Zero,
    Rgb<S, T>: core::ops::AddAssign<Rgb<S, T>> + Default,
{
    fn get_closest_centroid(rgb: &[Rgb<S, T>], centroids: &[Rgb<S, T>], indices: &mut Vec<u8>) {
        crate::kernels::assign(rgb, centroids, indices);
    }

    #[allow(clippy::cast_precision_loss)]
    fn recalculate_centroids(
        mut rng: &mut impl Rng,
        buf: &[Rgb<S, T>],
        centroids: &mut [Rgb<S, T>],
        indices: &[u8],
    ) {
        let sums = sum_clusters(buf, indices.iter().copied(), centroids.len());
        for (cent, (temp, counter)) in centroids.iter_mut().zip(sums) {
            if counter != 0 {
                *cent = temp / T::from_f64(counter as f64).unwrap();
            } else {
                *cent = Self::create_random(&mut rng);
            }
        }
    }

    fn check_loop(centroids: &[Rgb<S, T>], old_centroids: &[Rgb<S, T>]) -> f32 {
        centroids
            .iter()
            .zip(old_centroids)
            .map(|(c0, c1)| Self::difference(c0, c1))
            .sum()
    }

    #[inline]
    fn create_random(rng: &mut impl Rng) -> Rgb<S, T> {
        Rgb::<S, T>::new(
            T::from_f64(rng.random_range(0.0..=1.0)).unwrap(),
            T::from_f64(rng.random_range(0.0..=1.0)).unwrap(),
            T::from_f64(rng.random_range(0.0..=1.0)).unwrap(),
        )
    }

    #[inline]
    fn difference(c1: &Rgb<S, T>, c2: &Rgb<S, T>) -> f32 {
        let temp = *c1 - *c2;

        ((temp.red).powi(2) + (temp.green).powi(2) + (temp.blue).powi(2))
            .to_f32()
            .unwrap_or(f32::MAX)
    }
}

#[cfg(feature = "palette_color")]
impl<Wp, T> Hamerly for Lab<Wp, T>
where
    T: Float + FromPrimitive + Zero,
    Lab<Wp, T>: core::ops::AddAssign<Lab<Wp, T>> + Default,
{
    fn compute_half_distances(centers: &mut HamerlyCentroids<Self>) {
        compute_half_distances(centers);
    }

    fn get_closest_centroid_hamerly(
        buffer: &[Self],
        centers: &HamerlyCentroids<Self>,
        points: &mut [HamerlyPoint],
    ) {
        crate::kernels::assign_hamerly(buffer, centers, points);
    }

    #[allow(clippy::cast_precision_loss)]
    fn recalculate_centroids_hamerly(
        mut rng: &mut impl Rng,
        buf: &[Self],
        centers: &mut HamerlyCentroids<Self>,
        points: &[HamerlyPoint],
    ) {
        let sums = sum_clusters(
            buf,
            points.iter().map(|point| point.index),
            centers.centroids.len(),
        );
        for ((cent, delta), (temp, counter)) in centers
            .centroids
            .iter_mut()
            .zip(centers.deltas.iter_mut())
            .zip(sums)
        {
            let new_color = if counter != 0 {
                temp / T::from_f64(counter as f64).unwrap()
            } else {
                Self::create_random(&mut rng)
            };
            *delta = Self::difference(cent, &new_color).sqrt();
            *cent = new_color;
        }
    }

    fn update_bounds(centers: &HamerlyCentroids<Self>, points: &mut [HamerlyPoint]) {
        update_bounds(centers, points);
    }
}

#[cfg(feature = "palette_color")]
impl<S, T> Hamerly for Rgb<S, T>
where
    T: Float + FromPrimitive + Zero,
    Rgb<S, T>: core::ops::AddAssign<Rgb<S, T>> + Default,
{
    fn compute_half_distances(centers: &mut HamerlyCentroids<Self>) {
        compute_half_distances(centers);
    }

    fn get_closest_centroid_hamerly(
        buffer: &[Self],
        centers: &HamerlyCentroids<Self>,
        points: &mut [HamerlyPoint],
    ) {
        crate::kernels::assign_hamerly(buffer, centers, points);
    }

    #[allow(clippy::cast_precision_loss)]
    fn recalculate_centroids_hamerly(
        mut rng: &mut impl Rng,
        buf: &[Self],
        centers: &mut HamerlyCentroids<Self>,
        points: &[HamerlyPoint],
    ) {
        let sums = sum_clusters(
            buf,
            points.iter().map(|point| point.index),
            centers.centroids.len(),
        );
        for ((cent, delta), (temp, counter)) in centers
            .centroids
            .iter_mut()
            .zip(centers.deltas.iter_mut())
            .zip(sums)
        {
            let new_color = if counter != 0 {
                temp / T::from_f64(counter as f64).unwrap()
            } else {
                Self::create_random(&mut rng)
            };
            *delta = Self::difference(cent, &new_color).sqrt();
            *cent = new_color;
        }
    }

    fn update_bounds(centers: &HamerlyCentroids<Self>, points: &mut [HamerlyPoint]) {
        update_bounds(centers, points);
    }
}

/// A trait for mapping colors to their corresponding centroids.
#[cfg(feature = "palette_color")]
pub trait MapColor: Sized {
    /// Map pixel indices to each centroid for output buffer.
    fn map_indices_to_centroids(centroids: &[Self], indices: &[u8]) -> Vec<Self>;
}

#[cfg(feature = "palette_color")]
impl<Wp, T> MapColor for Lab<Wp, T>
where
    T: Copy,
{
    #[inline]
    fn map_indices_to_centroids(centroids: &[Self], indices: &[u8]) -> Vec<Self> {
        indices
            .iter()
            .map(|x| {
                *centroids
                    .get(*x as usize)
                    .unwrap_or_else(|| centroids.last().unwrap())
            })
            .collect()
    }
}

#[cfg(feature = "palette_color")]
impl<Wp, T> MapColor for palette::Laba<Wp, T>
where
    T: Copy,
{
    #[inline]
    fn map_indices_to_centroids(centroids: &[Self], indices: &[u8]) -> Vec<Self> {
        indices
            .iter()
            .map(|x| {
                *centroids
                    .get(*x as usize)
                    .unwrap_or_else(|| centroids.last().unwrap())
            })
            .collect()
    }
}

#[cfg(feature = "palette_color")]
impl<S, T> MapColor for Rgb<S, T>
where
    T: Copy,
{
    #[inline]
    fn map_indices_to_centroids(centroids: &[Self], indices: &[u8]) -> Vec<Self> {
        indices
            .iter()
            .map(|x| {
                *centroids
                    .get(*x as usize)
                    .unwrap_or_else(|| centroids.last().unwrap())
            })
            .collect()
    }
}

#[cfg(feature = "palette_color")]
impl<S, T> MapColor for Rgba<S, T>
where
    T: Copy,
{
    #[inline]
    fn map_indices_to_centroids(centroids: &[Self], indices: &[u8]) -> Vec<Self> {
        indices
            .iter()
            .map(|x| {
                *centroids
                    .get(*x as usize)
                    .unwrap_or_else(|| centroids.last().unwrap())
            })
            .collect()
    }
}

#[cfg(all(test, feature = "palette_color"))]
mod tests {
    use super::*;
    use palette::Srgb;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn opposite_centroid_movements_do_not_cancel() {
        let old_lab: [Lab; 2] = [Lab::new(0.0, 0.0, 0.0), Lab::new(1.0, 0.0, 0.0)];
        let new_lab = [Lab::new(0.25, 0.0, 0.0), Lab::new(0.75, 0.0, 0.0)];
        assert_eq!(Lab::check_loop(&new_lab, &old_lab), 0.125);
        let old_rgb = [Srgb::new(0.0, 0.0, 0.0), Srgb::new(1.0, 0.0, 0.0)];
        let new_rgb = [Srgb::new(0.25, 0.0, 0.0), Srgb::new(0.75, 0.0, 0.0)];
        assert_eq!(Srgb::check_loop(&new_rgb, &old_rgb), 0.125);
    }

    fn check_centroid_updates<C>(buf: &[C])
    where
        C: Hamerly
            + Copy
            + Default
            + PartialEq
            + core::fmt::Debug
            + core::ops::AddAssign
            + core::ops::Div<f32, Output = C>,
    {
        let indices = [2, 0, 2, 0, 255];
        let old_centroids = vec![buf[0]; 4];
        let mut expected = old_centroids.clone();
        let mut reference_rng = ChaCha8Rng::seed_from_u64(17);
        // Preserve accumulation order and random draws for empty clusters.
        for (index, centroid) in expected.iter_mut().enumerate() {
            let mut sum = C::default();
            let mut count = 0_u16;
            for (&label, &color) in indices.iter().zip(buf) {
                if label as usize == index {
                    sum += color;
                    count += 1;
                }
            }
            *centroid = if count > 0 {
                sum / f32::from(count)
            } else {
                C::create_random(&mut reference_rng)
            };
        }
        let next_random = reference_rng.next_u64();

        let mut rng = ChaCha8Rng::seed_from_u64(17);
        let mut actual = old_centroids.clone();
        C::recalculate_centroids(&mut rng, buf, &mut actual, &indices);
        assert_eq!(actual, expected);
        assert_eq!(rng.next_u64(), next_random);

        let points: Vec<_> = indices
            .iter()
            .map(|&index| HamerlyPoint {
                index,
                ..HamerlyPoint::new()
            })
            .collect();
        let mut centers = HamerlyCentroids::new(4);
        centers.centroids = old_centroids.clone();
        let mut rng = ChaCha8Rng::seed_from_u64(17);
        C::recalculate_centroids_hamerly(&mut rng, buf, &mut centers, &points);
        assert_eq!(centers.centroids, expected);
        assert_eq!(rng.next_u64(), next_random);
        for ((old, new), delta) in old_centroids.iter().zip(&expected).zip(centers.deltas) {
            assert_eq!(delta, C::difference(old, new).sqrt());
        }
    }

    #[test]
    fn centroid_updates_preserve_sums_and_empty_cluster_rng() {
        check_centroid_updates::<Lab>(&[
            Lab::new(0.1, 0.2, 0.3),
            Lab::new(0.8, 0.9, 0.7),
            Lab::new(0.2, 0.1, 0.5),
            Lab::new(0.4, 0.3, 0.9),
            Lab::new(0.0, 0.0, 0.0),
        ]);
        check_centroid_updates(&[
            Srgb::new(0.1, 0.2, 0.3),
            Srgb::new(0.8, 0.9, 0.7),
            Srgb::new(0.2, 0.1, 0.5),
            Srgb::new(0.4, 0.3, 0.9),
            Srgb::new(0.0, 0.0, 0.0),
        ]);
    }

    fn check_hamerly_distances_and_bounds<C: Hamerly + Copy>(colors: [C; 3]) {
        for centroids in [
            colors.to_vec(),
            vec![colors[0], colors[1], colors[1]],
            vec![colors[0]],
        ] {
            let mut centers = HamerlyCentroids::new(centroids.len());
            centers.centroids = centroids;
            C::compute_half_distances(&mut centers);
            for (i, color) in centers.centroids.iter().enumerate() {
                let nearest = centers
                    .centroids
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, other)| C::difference(color, other))
                    .fold(f32::MAX, f32::min);
                assert_eq!(centers.half_distances[i], nearest.sqrt() * 0.5);
            }

            for deltas in [[3.0, 1.0, 2.0], [3.0, 3.0, 1.0], [0.0; 3]] {
                centers.deltas = deltas[..centers.centroids.len()].to_vec();
                let mut points: Vec<_> = (0_u8..)
                    .take(centers.centroids.len())
                    .map(|index| HamerlyPoint {
                        index,
                        upper_bound: 5.0,
                        lower_bound: 10.0,
                    })
                    .collect();
                C::update_bounds(&centers, &mut points);
                for point in points {
                    let largest_other = centers
                        .deltas
                        .iter()
                        .enumerate()
                        .filter(|(index, _)| *index != point.index as usize)
                        .map(|(_, &delta)| delta)
                        .fold(0.0, f32::max);
                    assert_eq!(point.lower_bound, 10.0 - largest_other);
                    assert_eq!(point.upper_bound, 5.0 + deltas[point.index as usize]);
                }
            }
        }
    }

    #[test]
    fn hamerly_distances_and_bounds_match_exhaustive_calculation() {
        check_hamerly_distances_and_bounds::<Lab>([
            Lab::new(0.0, 0.0, 0.0),
            Lab::new(2.0, 0.0, 0.0),
            Lab::new(6.0, 0.0, 0.0),
        ]);
        check_hamerly_distances_and_bounds([
            Srgb::new(0.0, 0.0, 0.0),
            Srgb::new(0.2, 0.0, 0.0),
            Srgb::new(0.6, 0.0, 0.0),
        ]);
    }

    fn check_algorithms<C: Hamerly + Clone + PartialEq + core::fmt::Debug>(buf: &[C]) {
        for seed in 0..4 {
            for k in [1, 2, 4, 8] {
                let lloyd = crate::get_kmeans(k, 100, 0.0, false, buf, seed);
                let hamerly = crate::get_kmeans_hamerly(k, 100, 0.0, false, buf, seed);
                assert_eq!(lloyd.centroids, hamerly.centroids, "k={k}, seed={seed}");
                assert_eq!(lloyd.indices, hamerly.indices, "k={k}, seed={seed}");
                assert_eq!(lloyd.score, hamerly.score, "k={k}, seed={seed}");
            }
        }
    }

    #[test]
    fn lloyd_and_hamerly_agree_with_one_or_more_clusters_than_unique_colors() {
        let mut rng = ChaCha8Rng::seed_from_u64(43);
        let lab: Vec<Lab> = (0..5).map(|_| Lab::create_random(&mut rng)).collect();
        let rgb: Vec<Srgb> = (0..5).map(|_| Srgb::create_random(&mut rng)).collect();
        check_algorithms(&lab.repeat(3));
        check_algorithms(&rgb.repeat(3));
    }
}
