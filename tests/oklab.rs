#![cfg(feature = "palette_color")]

use kmeans_colors::{get_kmeans, get_kmeans_hamerly, Calculate, MapColor, Sort};
use palette::{FromColor, Oklab, Oklaba, Srgb};

#[test]
fn oklab_clusters_preserve_colors_and_population() {
    let red = Oklab::from_color(Srgb::new(1.0_f32, 0.0, 0.0));
    let green = Oklab::from_color(Srgb::new(0.0_f32, 1.0, 0.0));
    let points = [red, green, green, green];

    for seed in [0, 42, u64::MAX] {
        for cluster in [get_kmeans::<Oklab>, get_kmeans_hamerly::<Oklab>] {
            let result = cluster(2, 20, 0.0005, false, &points, seed);
            let colors = Oklab::sort_indexed_colors(&result.centroids, &result.indices);

            assert_eq!(colors.len(), 2);
            assert!(Oklab::difference(&colors[0].centroid, &red) < 1e-12);
            assert_eq!(colors[0].percentage, 0.25);
            assert!(Oklab::difference(&colors[1].centroid, &green) < 1e-12);
            assert_eq!(colors[1].percentage, 0.75);
            assert_eq!(Oklab::get_dominant_color(&colors), Some(colors[1].centroid));
            assert!(result.squared_error(&points) < 1e-12);
        }
    }
}

#[test]
fn oklab_average_and_movement_use_native_coordinate_scale() {
    let points = [Oklab::new(0.0_f64, 0.0, 0.0), Oklab::new(1.0, 0.0, 0.0)];
    let expected = Oklab::new(0.5, 0.0, 0.0);
    let result = get_kmeans(1, 20, 0.0005, false, &points, 42);

    assert_eq!(result.centroids, [expected]);
    assert_eq!(result.indices, [0, 0]);
    assert_eq!(result.squared_error(&points), 0.5);
    assert_eq!(Oklab::check_loop(&[expected, expected], &points), 0.5);
    assert_eq!(
        Oklab::map_indices_to_centroids(&result.centroids, &result.indices),
        [expected; 2]
    );

    let alpha = [
        Oklaba::new(0.0_f64, 0.0, 0.0, 0.5),
        Oklaba::new(1.0, 0.0, 0.0, 1.0),
    ];

    assert_eq!(
        Oklaba::map_indices_to_centroids(&alpha, &[1, 0]),
        [alpha[1], alpha[0]]
    );
}
