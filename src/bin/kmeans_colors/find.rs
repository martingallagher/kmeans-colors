use palette::cast::{AsComponents, ComponentsAs};
use palette::{white_point::D65, FromColor, IntoColor, Lab, Srgb, Srgba};

use crate::args::Command;
use crate::err::CliError;
use crate::filename::create_filename;
use crate::utils::{
    cached_srgba_to_lab, indexed_palette, map_opaque_pixels, parse_color, print_colors, save_image,
    save_image_alpha,
};
use kmeans_colors::{get_kmeans, get_kmeans_hamerly, Calculate, Kmeans, MapColor, Sort};

/// Find the image pixels which closest match the supplied colors and save that
/// image as output.
pub fn find_colors(
    Command::Find {
        input,
        colors,
        replace,
        max_iter,
        factor,
        runs,
        percentage,
        rgb,
        verbose,
        output,
        seed,
        transparent,
        fast_png,
    }: Command,
) -> Result<(), Box<dyn std::error::Error>> {
    // Print filename if multiple files and percentage is set
    let display_filename = (input.len() > 1) && (percentage);
    let converge = factor.unwrap_or(if !rgb { 5.0 } else { 0.0025 });

    let seed = seed.unwrap_or(0);

    // Cached results of Srgb<u8> -> Lab conversions; not cleared between runs
    let mut lab_cache = hashbrown::HashMap::default();
    // Vec of pixels converted to Lab; cleared and reused between runs
    let mut lab_pixels: Vec<Lab<D65, f32>> = Vec::new();
    // Vec of pixels converted to Srgb<f32>; cleared and reused between runs
    let mut rgb_pixels: Vec<Srgb<f32>> = Vec::new();

    // Default to Lab colors
    if !rgb {
        // Initialize user centroids
        let centroids: Vec<Lab<D65, f32>> = colors
            .iter()
            .map(|c| {
                parse_color(c.trim_start_matches('#')).map(|c| c.into_linear::<f32>().into_color())
            })
            .collect::<Result<_, CliError>>()?;

        for file in &input {
            if display_filename {
                println!("{}", file.to_string_lossy());
            }

            let img = image::open(file)?.into_rgba8();
            let (imgx, imgy) = img.dimensions();
            let img_vec: &[Srgba<u8>] = img.as_raw().components_as();

            lab_pixels.clear();

            if !transparent {
                cached_srgba_to_lab(img_vec.iter(), &mut lab_cache, &mut lab_pixels);
            } else {
                cached_srgba_to_lab(
                    img_vec.iter().filter(|x: &&Srgba<u8>| x.alpha == 255),
                    &mut lab_cache,
                    &mut lab_pixels,
                );
            }

            if !replace {
                let mut indices = Vec::with_capacity(lab_pixels.len());

                // We only need to do one pass of getting the closest colors to the
                // custom centroids
                Lab::<D65, f32>::get_closest_centroid(&lab_pixels, &centroids, &mut indices);

                if percentage {
                    let res = Lab::<D65, f32>::sort_indexed_colors(&centroids, &indices);
                    print_colors(percentage, &res)?;
                }

                if !transparent {
                    let rgb_centroids = &centroids
                        .iter()
                        .map(|&x| Srgb::from_linear(x.into_color()))
                        .collect::<Vec<Srgb<u8>>>();
                    let lab: Vec<Srgb<u8>> =
                        Srgb::map_indices_to_centroids(rgb_centroids, &indices);

                    save_image(
                        lab.as_components(),
                        imgx,
                        imgy,
                        &create_filename(&input, &output, "png", None, file)?,
                        false,
                        fast_png,
                    )?;
                } else {
                    let centroids = &centroids
                        .iter()
                        .map(|&x| {
                            Srgba::from(Srgb::<f32>::from_linear(x.into_color())).into_format()
                        })
                        .collect::<Vec<Srgba<u8>>>();
                    let rgba = map_opaque_pixels(img_vec, centroids, &indices);

                    save_image_alpha(
                        rgba.as_components(),
                        imgx,
                        imgy,
                        &create_filename(&input, &output, "png", None, file)?,
                        fast_png,
                    )?;
                }
            } else {
                // Replace the k-means colors case
                let mut result = Kmeans::new();
                let mut best_error = f32::INFINITY;
                let k = centroids.len();
                if k > 1 {
                    for i in 0..runs {
                        let run_result = get_kmeans_hamerly(
                            k,
                            max_iter,
                            converge,
                            verbose,
                            &lab_pixels,
                            seed + i as u64,
                        );
                        let error = run_result.squared_error(&lab_pixels);
                        if error < best_error {
                            best_error = error;
                            result = run_result;
                        }
                    }
                } else {
                    for i in 0..runs {
                        let run_result = get_kmeans(
                            k,
                            max_iter,
                            converge,
                            verbose,
                            &lab_pixels,
                            seed + i as u64,
                        );
                        let error = run_result.squared_error(&lab_pixels);
                        if error < best_error {
                            best_error = error;
                            result = run_result;
                        }
                    }
                }

                // We want to sort the user centroids based on the kmeans colors
                // sorted by luminosity using the u8 returned in `sorted`. This
                // corresponds to the index of the colors from darkest to lightest.
                // We replace the colors in `sorted` with our centroids for printing
                // purposes.
                let mut res =
                    Lab::<D65, f32>::sort_indexed_colors(&result.centroids, &result.indices);
                res.iter_mut()
                    .zip(&centroids)
                    .for_each(|(s, c)| s.centroid = *c);

                if percentage {
                    print_colors(percentage, &res)?;
                }

                let sorted = indexed_palette(&res, result.centroids.len());

                if !transparent {
                    let rgb_centroids = &sorted
                        .iter()
                        .map(|&x| Srgb::from_linear(x.into_color()))
                        .collect::<Vec<Srgb<u8>>>();
                    let rgb: Vec<Srgb<u8>> =
                        Srgb::map_indices_to_centroids(rgb_centroids, &result.indices);
                    save_image(
                        rgb.as_components(),
                        imgx,
                        imgy,
                        &create_filename(&input, &output, "png", None, file)?,
                        false,
                        fast_png,
                    )?;
                } else {
                    let mut indices = Vec::with_capacity(lab_pixels.len());
                    Lab::<D65, f32>::get_closest_centroid(
                        &lab_pixels,
                        &result.centroids,
                        &mut indices,
                    );

                    let centroids = &sorted
                        .iter()
                        .map(|&x| {
                            Srgba::from(Srgb::<f32>::from_linear(x.into_color())).into_format()
                        })
                        .collect::<Vec<Srgba<u8>>>();
                    let rgba = map_opaque_pixels(img_vec, centroids, &indices);

                    save_image_alpha(
                        rgba.as_components(),
                        imgx,
                        imgy,
                        &create_filename(&input, &output, "png", None, file)?,
                        fast_png,
                    )?;
                }
            }
        }

    // Rgb case
    } else {
        // Initialize user centroids
        let mut centroids: Vec<Srgb> = Vec::with_capacity(colors.len());
        for c in colors {
            centroids.push((parse_color(c.trim_start_matches('#'))?).into_format());
        }

        for file in &input {
            if display_filename {
                println!("{}", file.to_string_lossy());
            }
            let img = image::open(file)?.into_rgba8();
            let (imgx, imgy) = img.dimensions();
            let img_vec: &[Srgba<u8>] = img.as_raw().components_as();

            rgb_pixels.clear();

            if !transparent {
                rgb_pixels.extend(
                    img_vec
                        .iter()
                        .map(|x| Srgb::from_color(x.into_format::<_, f32>())),
                );
            } else {
                rgb_pixels.extend(
                    img_vec
                        .iter()
                        .filter(|x| x.alpha == 255)
                        .map(|x| Srgb::from_color(x.into_format::<_, f32>())),
                );
            }

            if !replace {
                let mut indices = Vec::with_capacity(rgb_pixels.len());

                // We only need to do one pass of getting the closest colors to the
                // custom centroids
                Srgb::get_closest_centroid(&rgb_pixels, &centroids, &mut indices);

                if percentage {
                    let res = Srgb::sort_indexed_colors(&centroids, &indices);
                    print_colors(percentage, &res)?;
                }

                if !transparent {
                    let rgb_centroids = &centroids
                        .iter()
                        .map(|x| x.into_format())
                        .collect::<Vec<Srgb<u8>>>();
                    let rgb: Vec<Srgb<u8>> =
                        Srgb::map_indices_to_centroids(rgb_centroids, &indices);

                    save_image(
                        rgb.as_components(),
                        imgx,
                        imgy,
                        &create_filename(&input, &output, "png", None, file)?,
                        false,
                        fast_png,
                    )?;
                } else {
                    let centroids = &centroids
                        .iter()
                        .map(|&x| Srgba::from(x).into_format())
                        .collect::<Vec<Srgba<u8>>>();
                    let rgba = map_opaque_pixels(img_vec, centroids, &indices);

                    save_image_alpha(
                        rgba.as_components(),
                        imgx,
                        imgy,
                        &create_filename(&input, &output, "png", None, file)?,
                        fast_png,
                    )?;
                }
            } else {
                // Replace the k-means colors case
                let mut result = Kmeans::new();
                let mut best_error = f32::INFINITY;
                let k = centroids.len();
                if k > 1 {
                    for i in 0..runs {
                        let run_result = get_kmeans_hamerly(
                            k,
                            max_iter,
                            converge,
                            verbose,
                            &rgb_pixels,
                            seed + i as u64,
                        );
                        let error = run_result.squared_error(&rgb_pixels);
                        if error < best_error {
                            best_error = error;
                            result = run_result;
                        }
                    }
                } else {
                    for i in 0..runs {
                        let run_result = get_kmeans(
                            k,
                            max_iter,
                            converge,
                            verbose,
                            &rgb_pixels,
                            seed + i as u64,
                        );
                        let error = run_result.squared_error(&rgb_pixels);
                        if error < best_error {
                            best_error = error;
                            result = run_result;
                        }
                    }
                }

                // We want to sort the user centroids based on the kmeans colors
                // sorted by luminosity using the u8 returned in `sorted`. This
                // corresponds to the index of the colors from darkest to lightest.
                // We replace the colors in `sorted` with our centroids for printing
                // purposes.
                let mut res = Srgb::sort_indexed_colors(&result.centroids, &result.indices);
                res.iter_mut()
                    .zip(&centroids)
                    .for_each(|(s, c)| s.centroid = *c);

                if percentage {
                    print_colors(percentage, &res)?;
                }

                let sorted = indexed_palette(&res, result.centroids.len());

                if !transparent {
                    let rgb_centroids = &sorted
                        .iter()
                        .map(|x| x.into_format())
                        .collect::<Vec<Srgb<u8>>>();
                    let rgb: Vec<Srgb<u8>> =
                        Srgb::map_indices_to_centroids(rgb_centroids, &result.indices);

                    save_image(
                        rgb.as_components(),
                        imgx,
                        imgy,
                        &create_filename(&input, &output, "png", None, file)?,
                        false,
                        fast_png,
                    )?;
                } else {
                    let mut indices = Vec::with_capacity(rgb_pixels.len());
                    Srgb::get_closest_centroid(&rgb_pixels, &result.centroids, &mut indices);

                    let centroids = &sorted
                        .iter()
                        .map(|&x| Srgba::from(x).into_format())
                        .collect::<Vec<Srgba<u8>>>();
                    let rgba = map_opaque_pixels(img_vec, centroids, &indices);

                    save_image_alpha(
                        rgba.as_components(),
                        imgx,
                        imgy,
                        &create_filename(&input, &output, "png", None, file)?,
                        fast_png,
                    )?;
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::args::Opt;
    use structopt::StructOpt;

    #[test]
    fn transparent_find_preserves_the_selected_color_space() {
        let dir = std::env::temp_dir().join(format!("kmeans-find-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let input = dir.join("input.png");
        let output = dir.join("output.png");
        let pixels = [
            255, 255, 0, 255, // Yellow is closer to green in Lab; RGB ties at red.
            255, 0, 0, 127, 0, 0, 255, 0, 255, 0, 0, 255, 0, 255, 0, 255,
        ];
        image::save_buffer(&input, &pixels, 5, 1, image::ColorType::Rgba8).unwrap();

        for rgb in [false, true] {
            for fast_png in [false, true] {
                let mut args = vec![
                    "kmeans_colors",
                    "find",
                    "--input",
                    input.to_str().unwrap(),
                    "--output",
                    output.to_str().unwrap(),
                    "--colors",
                    "ff0000,00ff00",
                    "--transparent",
                ];
                if rgb {
                    args.push("--rgb");
                }
                if fast_png {
                    args.push("--fast-png");
                }
                find_colors(Opt::from_iter(args).cmd.unwrap()).unwrap();
                let actual = image::open(&output).unwrap().into_rgba8();
                assert_eq!(
                    actual.get_pixel(0, 0).0,
                    if rgb {
                        [255, 0, 0, 255]
                    } else {
                        [0, 255, 0, 255]
                    },
                );
                assert_eq!(actual.get_pixel(1, 0).0, [0, 0, 0, 0]);
                assert_eq!(actual.get_pixel(2, 0).0, [0, 0, 0, 0]);
                assert_eq!(actual.get_pixel(3, 0).0, [255, 0, 0, 255]);
                assert_eq!(actual.get_pixel(4, 0).0, [0, 255, 0, 255]);
            }
        }
        std::fs::remove_dir_all(dir).unwrap();
    }
}
