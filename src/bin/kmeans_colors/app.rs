use crate::args::Opt;
use crate::filename::{create_filename, create_filename_palette};
use crate::utils::{
    cached_srgba_to_color, map_opaque_pixels, print_colors, save_image, save_image_alpha,
    save_palette,
};

use kmeans_colors::{
    get_diverse_palette, get_kmeans, get_kmeans_hamerly, Calculate, CentroidData, Hamerly, Kmeans,
    MapColor, Sort,
};
use palette::cast::{AsComponents, ComponentsAs};
use palette::{FromColor, IntoColor, Lab, LinSrgb, LinSrgba, Oklab, Srgb, Srgba};

pub fn run(opt: Opt) -> Result<(), Box<dyn std::error::Error>> {
    if opt.diverse && opt.k == 0 {
        return Err("A diverse palette requires at least one color.".into());
    }

    if opt.oklab || opt.diverse {
        run_with_space::<Oklab>(opt)
    } else {
        run_with_space::<Lab>(opt)
    }
}

fn run_with_space<C>(opt: Opt) -> Result<(), Box<dyn std::error::Error>>
where
    C: Hamerly
        + Sort
        + Copy
        + FromColor<LinSrgba>
        + FromColor<Oklab>
        + IntoColor<Srgb>
        + IntoColor<LinSrgb>,
    LinSrgba: FromColor<C>,
{
    if opt.input.is_empty() {
        eprintln!("No input files specified.")
    }

    let seed = opt.seed.unwrap_or(0);

    // Cached results of Srgb<u8> -> Lab/Oklab conversions; not cleared between runs
    let mut lab_cache = hashbrown::HashMap::default();
    // Vec of pixels converted to Lab/Oklab; cleared and reused between runs
    let mut lab_pixels: Vec<C> = Vec::new();
    // Vec of pixels converted to Srgb<f32>; cleared and reused between runs
    let mut rgb_pixels: Vec<Srgb<f32>> = Vec::new();

    for file in &opt.input {
        if opt.verbose {
            println!("{}", file.to_string_lossy());
        }
        let img = image::open(file)?.into_rgba8();
        let (imgx, imgy) = img.dimensions();
        let img_vec: &[Srgba<u8>] = img.as_raw().components_as();
        let converge = opt.factor.unwrap_or(if opt.oklab {
            0.0005
        } else if opt.rgb {
            0.0025
        } else {
            5.0
        });

        // Perceptual color spaces.
        if !opt.rgb {
            lab_pixels.clear();

            // Convert Srgb image buffer to the selected perceptual space for kmeans
            if !opt.diverse || !opt.no_file {
                if !opt.transparent {
                    cached_srgba_to_color(img_vec.iter(), &mut lab_cache, &mut lab_pixels);
                } else {
                    cached_srgba_to_color(
                        img_vec.iter().filter(|x: &&Srgba<u8>| x.alpha == 255),
                        &mut lab_cache,
                        &mut lab_pixels,
                    );
                }
            }

            let diverse_palette: Option<Vec<CentroidData<C>>> = if opt.diverse {
                let palette = get_diverse_palette(
                    usize::from(opt.k),
                    img_vec
                        .iter()
                        .filter(|pixel| !opt.transparent || pixel.alpha == 255)
                        .map(|pixel| pixel.color),
                );

                if palette.is_empty() {
                    return Err("No pixels available for palette extraction.".into());
                }

                Some(
                    palette
                        .into_iter()
                        .map(|color| CentroidData {
                            centroid: C::from_color(color.centroid),
                            percentage: color.percentage,
                            index: color.index,
                        })
                        .collect(),
                )
            } else {
                None
            };

            // Iterate over amount of runs keeping best results
            let mut result = Kmeans::new();
            let mut best_error = f32::INFINITY;

            if let Some(palette) = &diverse_palette {
                result.centroids = palette.iter().map(|color| color.centroid).collect();

                if !opt.no_file && !opt.transparent {
                    C::get_closest_centroid(&lab_pixels, &result.centroids, &mut result.indices);
                }
            } else if opt.k > 1 {
                for i in 0..opt.runs {
                    let run_result = get_kmeans_hamerly(
                        opt.k as usize,
                        opt.max_iter,
                        converge,
                        opt.verbose,
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
                for i in 0..opt.runs {
                    let run_result = get_kmeans(
                        opt.k as usize,
                        opt.max_iter,
                        converge,
                        opt.verbose,
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

            // Print and/or sort results, output to palette
            if opt.print || opt.percentage || opt.palette {
                let mut res = diverse_palette
                    .unwrap_or_else(|| C::sort_indexed_colors(&result.centroids, &result.indices));

                if opt.sort {
                    res.sort_unstable_by(|a, b| (b.percentage).total_cmp(&a.percentage));
                }

                if opt.print || opt.percentage {
                    print_colors(opt.percentage, &res)?;
                }

                if opt.palette {
                    save_palette(
                        &res,
                        opt.proportional,
                        opt.height,
                        opt.width,
                        &create_filename_palette(
                            &opt.input,
                            &opt.palette_output,
                            if opt.diverse {
                                "diverse"
                            } else if opt.oklab {
                                "oklab"
                            } else {
                                "lab"
                            },
                            Some(opt.k),
                            file,
                        )?,
                        opt.fast_png,
                    )?;
                }
            }

            // Don't allocate image buffer if no-file
            if opt.no_file {
                continue;
            }

            // Convert indexed colors to Srgb colors to output as final result
            if !opt.transparent {
                // Convert centroids to Srgb<u8> before mapping to buffer
                let centroids = &result
                    .centroids
                    .iter()
                    .map(|&x| Srgb::from_linear(x.into_color()))
                    .collect::<Vec<Srgb<u8>>>();
                let rgb: Vec<Srgb<u8>> = Srgb::map_indices_to_centroids(centroids, &result.indices);

                save_image(
                    rgb.as_components(),
                    imgx,
                    imgy,
                    &create_filename(&opt.input, &opt.output, &opt.extension, Some(opt.k), file)?,
                    false,
                    opt.fast_png,
                )?;
            } else {
                // Classify the retained opaque pixels against the final centroids.
                let mut indices = Vec::with_capacity(lab_pixels.len());

                C::get_closest_centroid(&lab_pixels, &result.centroids, &mut indices);

                let centroids = &result
                    .centroids
                    .iter()
                    .map(|&x| Srgba::<f32>::from_linear(LinSrgba::from_color(x)).into_format())
                    .collect::<Vec<Srgba<u8>>>();

                let rgba = map_opaque_pixels(img_vec, centroids, &indices);
                save_image_alpha(
                    rgba.as_components(),
                    imgx,
                    imgy,
                    &create_filename(&opt.input, &opt.output, &opt.extension, Some(opt.k), file)?,
                    opt.fast_png,
                )?;
            }
        } else {
            rgb_pixels.clear();

            // Read image buffer into Srgb format
            if !opt.transparent {
                rgb_pixels.extend(
                    img_vec
                        .iter()
                        .map(|x| Srgb::<f32>::from_color(x.into_format::<_, f32>())),
                );
            } else {
                rgb_pixels.extend(
                    img_vec
                        .iter()
                        .filter(|x| x.alpha == 255)
                        .map(|x| Srgb::<f32>::from_color(x.into_format::<_, f32>())),
                );
            }

            // Iterate over amount of runs keeping best results
            let mut result = Kmeans::new();
            let mut best_error = f32::INFINITY;
            if opt.k > 1 {
                for i in 0..opt.runs {
                    let run_result = get_kmeans_hamerly(
                        opt.k as usize,
                        opt.max_iter,
                        converge,
                        opt.verbose,
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
                for i in 0..opt.runs {
                    let run_result = get_kmeans(
                        opt.k as usize,
                        opt.max_iter,
                        converge,
                        opt.verbose,
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

            // Print and/or sort results, output to palette
            if opt.print || opt.percentage || opt.palette {
                let mut res = Srgb::sort_indexed_colors(&result.centroids, &result.indices);
                if opt.sort {
                    res.sort_unstable_by(|a, b| (b.percentage).total_cmp(&a.percentage));
                }

                if opt.print || opt.percentage {
                    print_colors(opt.percentage, &res)?;
                }

                if opt.palette {
                    save_palette(
                        &res,
                        opt.proportional,
                        opt.height,
                        opt.width,
                        &create_filename_palette(
                            &opt.input,
                            &opt.palette_output,
                            "rgb",
                            Some(opt.k),
                            file,
                        )?,
                        opt.fast_png,
                    )?;
                }
            }

            // Don't allocate image buffer if no-file
            if opt.no_file {
                continue;
            }

            // Convert indexed colors to Srgb colors to output as final result
            if !opt.transparent {
                // Pre-convert centroids into output format
                let centroids = &result
                    .centroids
                    .iter()
                    .map(|x| x.into_format())
                    .collect::<Vec<Srgb<u8>>>();
                let rgb: Vec<Srgb<u8>> = Srgb::map_indices_to_centroids(centroids, &result.indices);

                save_image(
                    rgb.as_components(),
                    imgx,
                    imgy,
                    &create_filename(&opt.input, &opt.output, &opt.extension, Some(opt.k), file)?,
                    false,
                    opt.fast_png,
                )?;
            } else {
                // Classify the retained opaque pixels against the final centroids.
                let mut indices = Vec::with_capacity(rgb_pixels.len());

                Srgb::get_closest_centroid(&rgb_pixels, &result.centroids, &mut indices);

                let centroids = &result
                    .centroids
                    .iter()
                    .map(|x| x.into_format().into())
                    .collect::<Vec<Srgba<u8>>>();

                let rgb = map_opaque_pixels(img_vec, centroids, &indices);
                save_image_alpha(
                    rgb.as_components(),
                    imgx,
                    imgy,
                    &create_filename(&opt.input, &opt.output, &opt.extension, Some(opt.k), file)?,
                    opt.fast_png,
                )?;
            }
        }
    }

    Ok(())
}
