use num_traits::{Float, FromPrimitive, Zero};
use palette::Oklab;
use rand::{Rng, RngExt};

use super::kmeans::{compute_half_distances, sum_clusters, update_bounds};
use crate::{Calculate, CentroidData, Hamerly, HamerlyCentroids, HamerlyPoint, MapColor, Sort};

impl<T> Calculate for Oklab<T>
where
    T: Float + FromPrimitive + Zero,
    Oklab<T>: core::ops::AddAssign<Oklab<T>> + Default,
{
    fn get_closest_centroid(lab: &[Oklab<T>], centroids: &[Oklab<T>], indices: &mut Vec<u8>) {
        crate::kernels::assign(lab, centroids, indices);
    }

    #[allow(clippy::cast_precision_loss)]
    fn recalculate_centroids(
        mut rng: &mut impl Rng,
        buf: &[Oklab<T>],
        centroids: &mut [Oklab<T>],
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

    fn check_loop(centroids: &[Oklab<T>], old_centroids: &[Oklab<T>]) -> f32 {
        centroids
            .iter()
            .zip(old_centroids)
            .map(|(c0, c1)| Self::difference(c0, c1))
            .sum()
    }

    #[inline]
    fn create_random(rng: &mut impl Rng) -> Oklab<T> {
        Oklab::<T>::new(
            T::from_f64(rng.random_range(0.0..=1.0)).unwrap(),
            T::from_f64(rng.random_range(-0.4..=0.4)).unwrap(),
            T::from_f64(rng.random_range(-0.4..=0.4)).unwrap(),
        )
    }

    #[inline]
    fn difference(c1: &Oklab<T>, c2: &Oklab<T>) -> f32 {
        let temp = *c1 - *c2;

        ((temp.l).powi(2) + (temp.a).powi(2) + (temp.b).powi(2))
            .to_f32()
            .unwrap_or(f32::MAX)
    }
}

impl<T> Hamerly for Oklab<T>
where
    T: Float + FromPrimitive + Zero,
    Oklab<T>: core::ops::AddAssign<Oklab<T>> + Default,
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

impl<T> Sort for Oklab<T>
where
    T: Float + FromPrimitive + Zero,
    Oklab<T>: core::ops::AddAssign<Oklab<T>> + Default,
{
    fn get_dominant_color(data: &[CentroidData<Self>]) -> Option<Self> {
        data.iter()
            .max_by(|a, b| (a.percentage).partial_cmp(&b.percentage).unwrap())
            .map(|res| res.centroid)
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn sort_indexed_colors(centroids: &[Self], indices: &[u8]) -> Vec<CentroidData<Self>> {
        let mut counts = [0_u64; 256];

        for &index in indices {
            counts[index as usize] += 1;
        }

        assert!(!indices.is_empty());

        let mut colors: Vec<_> = centroids
            .iter()
            .enumerate()
            .filter(|(index, _)| counts[*index as u8 as usize] != 0)
            .map(|(index, &centroid)| CentroidData {
                centroid,
                percentage: counts[index as u8 as usize] as f32 / indices.len() as f32,
                index: index as u8,
            })
            .collect();

        colors.sort_unstable_by(|a, b| a.centroid.l.partial_cmp(&b.centroid.l).unwrap());

        colors
    }
}

impl<T> MapColor for Oklab<T>
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

impl<T> MapColor for palette::Oklaba<T>
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
