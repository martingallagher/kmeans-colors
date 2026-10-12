#![cfg(feature = "palette_color")]

use kmeans_colors::{get_diverse_palette, get_kmeans, Calculate, CentroidData};
use palette::{FromColor, Oklab, Srgb};

fn rgb(color: &CentroidData<Oklab>) -> Srgb<u8> {
    Srgb::from_color(color.centroid).into_format()
}

#[test]
fn one_red_pixel_survives_ten_thousand_background_pixels() {
    let red = Srgb::new(255, 0, 0);
    let mut pixels = Vec::new();

    for gray in [112, 120, 128, 136] {
        pixels.extend(std::iter::repeat_n(Srgb::new(gray, gray, gray), 2_500));
    }

    pixels.push(red);

    let palette = get_diverse_palette(3, pixels.iter().copied());
    let accent = palette.iter().find(|color| rgb(color) == red).unwrap();
    let points: Vec<Oklab> = pixels
        .iter()
        .map(|pixel| Oklab::from_color(pixel.into_format::<f32>().into_linear()))
        .collect();
    let ordinary = get_kmeans(3, 30, 1e-9, false, &points, 42);
    let red_lab = Oklab::from_color(red.into_format::<f32>().into_linear());

    assert_eq!(palette.len(), 3);
    assert!((accent.percentage - 1.0 / 10_001.0).abs() < 1e-8);
    assert!(ordinary
        .centroids
        .iter()
        .all(|color| Oklab::difference(color, &red_lab) > 0.01));
    assert!((palette.iter().map(|color| color.percentage).sum::<f32>() - 1.0).abs() < 1e-6);

    pixels.reverse();

    let reversed = get_diverse_palette(3, pixels);

    for (a, b) in palette.iter().zip(reversed) {
        assert_eq!(a.centroid, b.centroid);
        assert_eq!(a.percentage, b.percentage);
        assert_eq!(a.index, b.index);
    }
}

#[test]
fn percentages_use_original_counts_and_colors_are_observed() {
    let red = Srgb::new(255, 0, 0);
    let blue = Srgb::new(0, 0, 255);
    let pixels = std::iter::repeat_n(blue, 99).chain(std::iter::once(red));
    let palette = get_diverse_palette(8, pixels);

    assert_eq!(palette.len(), 2);
    assert_eq!(rgb(&palette[0]), blue);
    assert_eq!(rgb(&palette[1]), red);
    assert_eq!(palette[0].percentage, 0.99);
    assert_eq!(palette[1].percentage, 0.01);
    assert_eq!(palette[0].index, 0);
    assert_eq!(palette[1].index, 1);
}

#[test]
fn empty_zero_single_color_and_shared_bins() {
    let black = Srgb::new(0, 0, 0);

    assert!(get_diverse_palette(4, []).is_empty());
    assert!(get_diverse_palette(0, [black]).is_empty());

    let palette = get_diverse_palette(256, [black; 10]);

    assert_eq!(palette.len(), 1);
    assert_eq!(rgb(&palette[0]), black);
    assert_eq!(palette[0].percentage, 1.0);

    let pixels = [Srgb::new(0, 0, 0), Srgb::new(3, 4, 3), Srgb::new(7, 7, 7)];
    let palette = get_diverse_palette(3, pixels);

    assert_eq!(palette.len(), 1);
    assert_eq!(rgb(&palette[0]), pixels[1]);
    assert_eq!(palette[0].percentage, 1.0);
}

#[test]
#[should_panic(expected = "palette size must be at most 256")]
fn rejects_palette_sizes_that_do_not_fit_indices() {
    get_diverse_palette(257, [Srgb::new(0, 0, 0)]);
}

#[test]
fn supports_all_256_palette_indices() {
    let pixels =
        (0..17).flat_map(|red| (0..17).map(move |green| Srgb::new(red * 8, green * 8, 128)));
    let palette = get_diverse_palette(256, pixels);

    assert_eq!(palette.len(), 256);
    assert_eq!(palette.last().unwrap().index, 255);
    assert!(
        (palette
            .iter()
            .map(|color| f64::from(color.percentage))
            .sum::<f64>()
            - 1.0)
            .abs()
            < 1e-6
    );
}
