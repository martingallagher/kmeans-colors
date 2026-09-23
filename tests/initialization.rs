#![cfg(feature = "palette_color")]

use kmeans_colors::{init_plus_plus, Kmeans};
use palette::Lab;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn points() -> Vec<Lab> {
    (0..24)
        .map(|i| {
            Lab::new(
                (i * 17 % 101) as f32,
                (i * 37 % 255) as f32 - 128.0,
                (i * 53 % 255) as f32 - 128.0,
            )
        })
        .collect()
}

#[test]
fn seeded_initialization_preserves_center_choices() {
    let points = points();
    // Recorded with the original initialization and rand 0.9 / ChaCha8Rng.
    for (seed, expected) in [
        (0, [15, 18, 19, 11, 12, 16, 14, 0]),
        (42, [5, 17, 0, 22, 19, 10, 9, 13]),
        (u64::MAX, [21, 17, 13, 9, 0, 7, 12, 10]),
    ] {
        let mut centroids = Vec::new();
        init_plus_plus(
            8,
            &mut ChaCha8Rng::seed_from_u64(seed),
            &points,
            &mut centroids,
        );

        let expected: Vec<_> = expected.iter().map(|&index| points[index]).collect();
        assert_eq!(centroids, expected, "seed {seed}");
    }
}

#[test]
fn initialization_accounts_for_prefilled_centroids() {
    let points = points();
    let mut centroids = vec![points[0], points[7]];
    init_plus_plus(
        4,
        &mut ChaCha8Rng::seed_from_u64(42),
        &points,
        &mut centroids,
    );

    // The original API adds k centers after those supplied by the caller.
    let expected: Vec<_> = [0, 7, 5, 18, 6, 23]
        .iter()
        .map(|&index| points[index])
        .collect();
    assert_eq!(centroids, expected);
}

#[test]
fn initialization_stops_when_all_distinct_points_are_centers() {
    let points = points();
    let duplicates = [points[0], points[0], points[1], points[1], points[2]];
    let mut centroids = Vec::new();
    init_plus_plus(
        12,
        &mut ChaCha8Rng::seed_from_u64(17),
        &duplicates,
        &mut centroids,
    );

    assert_eq!(centroids, [points[1], points[2], points[0]]);
}

#[test]
fn zero_centers_leaves_existing_centroids_unchanged() {
    let mut centroids: Vec<Lab> = vec![Lab::new(10.0, 20.0, 30.0)];
    init_plus_plus(0, &mut ChaCha8Rng::seed_from_u64(0), &[], &mut centroids);

    assert_eq!(centroids, [Lab::new(10.0, 20.0, 30.0)]);
}

#[test]
fn squared_error_compares_quality_independently_of_movement() {
    let points: Vec<Lab> = [0.0, 2.0, 8.0, 10.0]
        .iter()
        .map(|&l| Lab::new(l, 0.0, 0.0))
        .collect();
    let better = Kmeans {
        score: 1.0,
        centroids: vec![Lab::new(1.0, 0.0, 0.0), Lab::new(9.0, 0.0, 0.0)],
        indices: vec![0, 0, 1, 1],
    };
    let worse = Kmeans {
        score: 0.0,
        centroids: vec![Lab::new(0.0, 0.0, 0.0), Lab::new(10.0, 0.0, 0.0)],
        indices: better.indices.clone(),
    };

    assert!(better.score > worse.score);
    assert_eq!(better.squared_error(&points), 4.0);
    assert_eq!(worse.squared_error(&points), 8.0);
}
