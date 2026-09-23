use crate::sort::{CentroidData, Sort};

#[cfg(feature = "palette_color")]
use num_traits::{Float, FromPrimitive, Zero};
#[cfg(feature = "palette_color")]
use palette::{luma::Luma, rgb::Rgb, IntoColor, Lab};

#[cfg(feature = "palette_color")]
impl<Wp, T> Sort for Lab<Wp, T>
where
    T: Float + FromPrimitive + Zero,
    Lab<Wp, T>: core::ops::AddAssign<Lab<Wp, T>> + Default,
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

#[cfg(feature = "palette_color")]
impl<S, T> Sort for Rgb<S, T>
where
    T: Float + FromPrimitive + Zero,
    Rgb<S, T>: core::ops::AddAssign<Rgb<S, T>> + IntoColor<Luma<S, T>> + Default,
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
            .map(|(index, &centroid)| {
                let luma: Luma<S, T> = centroid.into_color();
                (
                    CentroidData {
                        centroid,
                        percentage: counts[index as u8 as usize] as f32 / indices.len() as f32,
                        index: index as u8,
                    },
                    luma.luma,
                )
            })
            .collect();
        colors.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        colors.into_iter().map(|(color, _)| color).collect()
    }
}

#[cfg(all(test, feature = "palette_color"))]
mod tests {
    use crate::{CentroidData, Sort};
    use palette::{Lab, Srgb};

    #[test]
    fn sparse_labels_keep_original_indices_and_percentages() {
        let lab: [Lab; 4] = [
            Lab::new(100.0, 0.0, 0.0),
            Lab::new(20.0, 0.0, 0.0),
            Lab::new(0.0, 0.0, 0.0),
            Lab::new(50.0, 0.0, 0.0),
        ];
        let rgb = [
            Srgb::new(1.0, 1.0, 1.0),
            Srgb::new(0.2, 0.2, 0.2),
            Srgb::new(0.0, 0.0, 0.0),
            Srgb::new(0.5, 0.5, 0.5),
        ];
        let indices = [0, 3, 3, 3, 2, 2, 2, 2];
        let expected = [(2, 0.5), (3, 0.375), (0, 0.125)];
        let sorted = Lab::sort_indexed_colors(&lab, &indices);
        assert_eq!(sorted.len(), expected.len());
        for (color, &(index, percentage)) in sorted.iter().zip(&expected) {
            assert_eq!(color.index, index);
            assert_eq!(color.percentage, percentage);
            assert_eq!(color.centroid, lab[index as usize]);
        }
        let sorted = Srgb::sort_indexed_colors(&rgb, &indices);
        assert_eq!(sorted.len(), expected.len());
        for (color, &(index, percentage)) in sorted.iter().zip(&expected) {
            assert_eq!(color.index, index);
            assert_eq!(color.percentage, percentage);
            assert_eq!(color.centroid, rgb[index as usize]);
        }

        let centroids = vec![Srgb::new(0.0, 0.0, 0.0); 256];
        let sorted = Srgb::sort_indexed_colors(&centroids, &[255, 255]);
        assert_eq!(sorted.len(), 1);
        assert_eq!(sorted[0].index, 255);
        assert_eq!(sorted[0].percentage, 1.0);
    }

    #[test]
    fn dominant_color() {
        let res = vec![
            CentroidData::<Srgb> {
                centroid: Srgb::new(0.0, 0.0, 0.0),
                percentage: 0.5,
                index: 0,
            },
            CentroidData::<Srgb> {
                centroid: Srgb::new(0.5, 0.5, 0.5),
                percentage: 0.80,
                index: 0,
            },
            CentroidData::<Srgb> {
                centroid: Srgb::new(1.0, 1.0, 1.0),
                percentage: 0.15,
                index: 0,
            },
        ];
        assert_eq!(
            Srgb::get_dominant_color(&res).unwrap(),
            Srgb::new(0.5, 0.5, 0.5)
        );
    }
}
