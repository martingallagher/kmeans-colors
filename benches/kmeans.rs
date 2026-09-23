//! Run with `cargo bench --bench kmeans` from the repository root.
use std::{hint::black_box, time::Instant};

use kmeans_colors::{get_kmeans, get_kmeans_hamerly, init_plus_plus, Sort};
use palette::{IntoColor, Lab, Srgb};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn median_ms(mut run: impl FnMut()) -> f64 {
    run();
    let mut samples = [0.0; 5];
    for sample in &mut samples {
        let start = Instant::now();
        run();
        *sample = start.elapsed().as_secs_f64() * 1000.0;
    }
    samples.sort_by(f64::total_cmp);
    samples[2]
}

fn main() {
    println!("image,pixels,k,initialization_ms,lloyd_ms,hamerly_ms,sort_ms");
    for file in ["gfx/flowers.jpg", "gfx/mountains.jpg"] {
        let image = image::open(file).unwrap().to_rgb8();
        let pixels: Vec<Lab> = image
            .pixels()
            .map(|pixel| {
                Srgb::new(pixel[0], pixel[1], pixel[2])
                    .into_linear::<f32>()
                    .into_color()
            })
            .collect();
        for k in [8, 20, 64] {
            let initialization = median_ms(|| {
                let mut centroids = Vec::with_capacity(k);
                init_plus_plus(
                    k,
                    &mut ChaCha8Rng::seed_from_u64(42),
                    black_box(&pixels),
                    &mut centroids,
                );
                black_box(centroids);
            });
            let lloyd = median_ms(|| {
                black_box(get_kmeans(k, 20, 5.0, false, black_box(&pixels), 42));
            });
            let hamerly = median_ms(|| {
                black_box(get_kmeans_hamerly(
                    k,
                    20,
                    5.0,
                    false,
                    black_box(&pixels),
                    42,
                ));
            });
            let result = get_kmeans(k, 20, 5.0, false, &pixels, 42);
            let sort = median_ms(|| {
                black_box(Lab::sort_indexed_colors(
                    black_box(&result.centroids),
                    black_box(&result.indices),
                ));
            });
            println!(
                "{file},{},{k},{initialization:.3},{lloyd:.3},{hamerly:.3},{sort:.3}",
                pixels.len()
            );
        }
    }
}
