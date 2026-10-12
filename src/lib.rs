//! Calculate the k-means of a set of data.
//!
//! # Overview
//!
//! This crate provides traits for implementing and calculating a k-means
//! clustering algorithm. The original implementation of this library was
//! created for finding k-means colors in image buffers. Applications of crate
//! functionality can be seen on the [README page][readme].
//!
//! [readme]: https://github.com/okaneco/kmeans-colors/blob/master/README.md
//!
//! When using the library, set `default-features = false` in the Cargo.toml to
//! avoid bringing in the binary dependencies. If working with colors,
//! implementations have been provided for the [`palette`][palette] `Lab`,
//! `Oklab`, and `Rgb` color types behind the `palette_color` feature.
//!
//! Enable the optional `simd` feature for accelerated color assignment,
//! k-means++ initialization, and Hamerly searches using `fearless_simd`. It is
//! enabled in default builds and requires Rust 1.89 or newer. Existing generic
//! color types and seeded initialization are preserved.
//!
//! The binary located in `src/bin/kmeans_colors` shows examples of crate
//! usage.
//!
//! [palette]: https://github.com/Ogeon/palette/
//!
//! ## The `Calculate` trait
//! k-means calculations can be provided for other data types by implementing
//! the [`Calculate`](trait.Calculate.html) trait. Further,
//! [`Hamerly`](trait.Hamerly.html) can be implemented to enable use of the
//! Hamerly optimization and [`get_kmeans_hamerly`][hamerly]. See the `Lab` and
//! `Rgb` implementations in [`colors/kmeans.rs`][kmeans] for examples. These
//! implementations can be used as groundwork for implementing with other types
//! and should not require much modification beyond the distance calculations.
//!
//! [hamerly]: fn.get_kmeans_hamerly.html
//! [kmeans]: ../src/kmeans_colors/colors/kmeans.rs.html#9
//!
//! ## Calculating k-means with `palette_color`
//!
//! The `palette_color` feature provides implementations of the `Calculate`
//! trait for the `Lab`, `Oklab`, and `Rgb` color spaces. Each space has
//! advantages and drawbacks due to the characteristics of the color space.
//!
//! `Lab` and `Oklab` use perceptual coordinates, while `Rgb` clusters RGB
//! channel values directly. At `k=1`, each computes the average in its own
//! space, so the resulting RGB colors can differ substantially.
//!
//! Note: If k-means calculation is taking too long, try scaling down the
//! image size. A full-size image is not required for calculating the color
//! palette or dominant color.
//!
//! ### Calculating k-means
//!
//! A basic workflow consists of reading a pixel buffer in, converting it into a
//! flat array, then using that array with the k-means functions. The following
//! example converts an array of `u8` into `Lab` colors then finds the k-means.
//!
//! ```
//! use palette::cast::{from_component_slice, into_component_slice};
//! use palette::{FromColor, IntoColor, Lab, Srgb};
//! use kmeans_colors::{get_kmeans, Calculate, Kmeans, MapColor, Sort};
//!
//! // An image buffer of one black pixel and one white pixel
//! let img_vec = [0u8, 0, 0, 255, 255, 255];
//!
//! # let runs = 1;
//! # let k = 1;
//! # let max_iter = 20;
//! # let converge = 8.0;
//! # let verbose = false;
//! # let seed = 0;
//! // Convert RGB [u8] buffer to Lab for k-means
//! let lab: Vec<Lab> = from_component_slice::<Srgb<u8>>(&img_vec)
//!     .iter()
//!     .map(|x| x.into_linear().into_color())
//!     .collect();
//!
//! // Iterate over the runs, keep the best results
//! let mut result = Kmeans::new();
//! let mut best_error = f32::MAX;
//! for i in 0..runs {
//!     let run_result = get_kmeans(
//!         k,
//!         max_iter,
//!         converge,
//!         verbose,
//!         &lab,
//!         seed + i as u64,
//!     );
//!     let error = run_result.squared_error(&lab);
//!     if error < best_error {
//!         best_error = error;
//!         result = run_result;
//!     }
//! }
//!
//! // Convert indexed colors back to Srgb<u8> for output
//! let rgb = &result.centroids
//!     .iter()
//!     .map(|&x| Srgb::from_linear(x.into_color()))
//!     .collect::<Vec<Srgb<u8>>>();
//! let buffer = Srgb::map_indices_to_centroids(&rgb, &result.indices);
//! # assert_eq!(into_component_slice(&buffer), [119, 119, 119, 119, 119, 119]);
//! # // Test get_kmeans_hamerly
//! # let mut result = Kmeans::new();
//! # let mut best_error = f32::MAX;
//! # for i in 0..runs {
//! #     let run_result = kmeans_colors::get_kmeans_hamerly(
//! #         k,
//! #         max_iter,
//! #         converge,
//! #         verbose,
//! #         &lab,
//! #         seed + i as u64,
//! #     );
//! #     let error = run_result.squared_error(&lab);
//! #     if error < best_error {
//! #         best_error = error;
//! #         result = run_result;
//! #     }
//! # }
//! # // Convert indexed colors back to Srgb<u8> for output
//! # let rgb = &result.centroids
//! #     .iter()
//! #     .map(|&x| Srgb::from_linear(x.into_color()))
//! #     .collect::<Vec<Srgb<u8>>>();
//! # let buffer = Srgb::map_indices_to_centroids(&rgb, &result.indices);
//! # assert_eq!(into_component_slice(&buffer), [119, 119, 119, 119, 119, 119]);
//! ```
//!
//! k-means++ is used for centroid initialization. Because the initialization is
//! random, the k-means calculation may be run multiple times to assure that
//! the best result has been found. The algorithm can find itself in a
//! sub-optimal result due to initial centroids. Compare runs using
//! [`Kmeans::squared_error`], which measures distances from points to their
//! assigned centers. The convergence `score` only measures centroid movement
//! and does not indicate clustering quality.
//!
//! The binary uses `8` as the default `k`. The iteration limit is set to `20`.
//! The convergence factor defaults to `5.0` for `Lab`, `0.0005` for `Oklab`,
//! and `0.0025` for `Rgb`.
//! The number of runs defaults to `3` for one of the binary subcommands.
//! If the results do not appear correct, raise the iteration limit as
//! convergence was probably not met.
//!
//! ### Oklab
//!
//! Oklab supports both Lloyd's and Hamerly's algorithms, SIMD, lightness
//! sorting, and population reporting. It uses native coordinates (`L` is
//! `0..=1` for sRGB), so Lab convergence thresholds should not be reused.
//! `0.0005` is the CLI's starting threshold; lower it for tighter convergence.
//!
//! ```
//! use kmeans_colors::{get_kmeans_hamerly, Sort};
//! use palette::{IntoColor, Oklab, Srgb};
//!
//! let pixels = [Srgb::new(255u8, 0, 0), Srgb::new(0, 255, 0)];
//! let colors: Vec<Oklab> = pixels.iter()
//!     .map(|pixel| pixel.into_linear::<f32>().into_color())
//!     .collect();
//! let result = get_kmeans_hamerly(2, 20, 0.0005, false, &colors, 42);
//! let palette = Oklab::sort_indexed_colors(&result.centroids, &result.indices);
//! assert_eq!(palette.len(), 2);
//! assert!(palette.iter().all(|color| color.percentage == 0.5));
//! ```
//!
//! ### Getting the dominant color
//!
//! After k-means calculation, the dominant color can be found by sorting the
//! results and taking the centroid of the first item. The
//! [`sort_indexed_colors`][sort] function sorts the colors from darkest to
//! lightest and returns an array of [`CentroidData`](struct.CentroidData.html).
//!
//! [sort]: trait.Sort.html#tymethod.sort_indexed_colors
//! ```
//! # use palette::{cast::from_component_slice, FromColor, IntoColor, Lab, Srgb};
//! # use kmeans_colors::{get_kmeans, Kmeans};
//! use kmeans_colors::Sort;
//!
//! # let img_vec = [0u8, 0, 0, 255, 255, 255];
//! # let runs = 1;
//! # let k = 1;
//! # let max_iter = 20;
//! # let converge = 8.0;
//! # let verbose = false;
//! # let seed = 0;
//! # let lab: Vec<Lab> = from_component_slice::<Srgb<u8>>(&img_vec)
//! #    .iter()
//! #    .map(|x| x.into_format().into_color())
//! #    .collect();
//! # let mut result = Kmeans::new();
//! # let mut best_error = f32::MAX;
//! # for i in 0..runs {
//! #     let run_result = get_kmeans(
//! #         k,
//! #         max_iter,
//! #         converge,
//! #         verbose,
//! #         &lab,
//! #         seed + i as u64,
//! #     );
//! #     let error = run_result.squared_error(&lab);
//! #     if error < best_error {
//! #         best_error = error;
//! #         result = run_result;
//! #     }
//! # }
//! // Using the results from the previous example, process the centroid data
//! let mut res = Lab::sort_indexed_colors(&result.centroids, &result.indices);
//!
//! // We can find the dominant color directly
//! let dominant_color = Lab::get_dominant_color(&res);
//! # assert_eq!(
//! #    Srgb::from_color(dominant_color.unwrap()).into_format::<u8>(),
//! #    Srgb::new(119u8, 119, 119)
//! # );
//!
//! // Or we can manually sort the vec by percentage, and the most appearing
//! // color will be the first element
//! res.sort_unstable_by(|a, b| (b.percentage).total_cmp(&a.percentage));
//! let dominant_color = res.first().unwrap().centroid;
//! ```
#![forbid(
    absolute_paths_not_starting_with_crate,
    missing_docs,
    non_ascii_idents,
    noop_method_call,
    rust_2018_idioms,
    unsafe_code,
    unused_results
)]
#![warn(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

#[cfg(feature = "palette_color")]
mod colors;

mod kernels;
mod kmeans;
mod plus_plus;
mod sort;

#[cfg(feature = "palette_color")]
pub use colors::{get_diverse_palette, MapColor};

pub use kmeans::{
    get_kmeans, get_kmeans_hamerly, Calculate, Hamerly, HamerlyCentroids, HamerlyPoint, Kmeans,
};
pub use plus_plus::init_plus_plus;
pub use sort::{CentroidData, Sort};
