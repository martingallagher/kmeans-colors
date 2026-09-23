use std::error::Error;
use std::fmt::Write;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::str::FromStr;

use image::ImageEncoder;
use palette::{white_point::D65, IntoColor, Lab, Srgb, Srgba};

use crate::err::CliError;
use kmeans_colors::{Calculate, CentroidData};

/// Parse hex string to Rgb color.
pub fn parse_color(c: &str) -> Result<Srgb<u8>, CliError> {
    Srgb::from_str(c).map_err(|_| {
        eprintln!("Invalid color: {c}");
        CliError::InvalidHex
    })
}

/// Prints colors and percentage of their appearance in an image buffer.
pub fn print_colors<C: Calculate + Copy + IntoColor<Srgb>>(
    show_percentage: bool,
    colors: &[CentroidData<C>],
) -> Result<(), Box<dyn Error>> {
    let mut col = String::new();
    let mut freq = String::new();
    if let Some((last, elements)) = colors.split_last() {
        for elem in elements {
            write!(
                &mut col,
                "{:x},",
                elem.centroid.into_color().into_format::<u8>()
            )?;
            write!(&mut freq, "{:0.4},", elem.percentage)?;
        }
        writeln!(
            &mut col,
            "{:x}",
            last.centroid.into_color().into_format::<u8>()
        )?;
        writeln!(&mut freq, "{:0.4}", last.percentage)?;
    }
    print!("{}", col);
    if show_percentage {
        print!("{}", freq);
    }

    Ok(())
}

/// Saves image buffer to file.
pub fn save_image(
    imgbuf: &[u8],
    imgx: u32,
    imgy: u32,
    title: &Path,
    palette: bool,
    fast_png: bool,
) -> Result<(), Box<dyn Error>> {
    let mut w = BufWriter::new(File::create(title)?);
    if title.extension().unwrap() == "png" {
        // If file is a palette, use Adaptive filtering to save more space
        use image::codecs::png::FilterType::{Adaptive, NoFilter};
        let encoder = image::codecs::png::PngEncoder::new_with_quality(
            w,
            if fast_png {
                image::codecs::png::CompressionType::Fast
            } else {
                image::codecs::png::CompressionType::Best
            },
            if palette { Adaptive } else { NoFilter },
        );

        // Clean up if file is created but there's a problem writing to it
        match encoder.write_image(imgbuf, imgx, imgy, image::ExtendedColorType::Rgb8) {
            Ok(_) => {}
            Err(err) => {
                eprintln!("Error: {}.", err);
                std::fs::remove_file(title)?;
            }
        }
    } else {
        let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut w, 90);

        match encoder.encode(imgbuf, imgx, imgy, image::ExtendedColorType::Rgb8) {
            Ok(_) => {}
            Err(err) => {
                eprintln!("Error: {}.", err);
                std::fs::remove_file(title)?;
            }
        }
    };

    Ok(())
}

/// Saves transparent image buffer to file.
pub fn save_image_alpha(
    imgbuf: &[u8],
    imgx: u32,
    imgy: u32,
    title: &Path,
    fast_png: bool,
) -> Result<(), Box<dyn Error>> {
    let mut w = BufWriter::new(File::create(title)?);
    if title.extension().unwrap() == "png" {
        let encoder = image::codecs::png::PngEncoder::new_with_quality(
            w,
            if fast_png {
                image::codecs::png::CompressionType::Fast
            } else {
                image::codecs::png::CompressionType::Best
            },
            image::codecs::png::FilterType::NoFilter,
        );

        // Clean up if file is created but there's a problem writing to it
        match encoder.write_image(imgbuf, imgx, imgy, image::ExtendedColorType::Rgba8) {
            Ok(_) => {}
            Err(err) => {
                eprintln!("Error: {}.", err);
                std::fs::remove_file(title)?;
            }
        }
    } else {
        let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut w, 90);

        match encoder.encode(imgbuf, imgx, imgy, image::ExtendedColorType::Rgba8) {
            Ok(_) => {}
            Err(err) => {
                eprintln!("Error: {}.", err);
                std::fs::remove_file(title)?;
            }
        }
    };

    Ok(())
}

/// Save palette image file.
pub fn save_palette<C: Calculate + Copy + IntoColor<Srgb>>(
    res: &[CentroidData<C>],
    proportional: bool,
    height: u32,
    width: Option<u32>,
    title: &Path,
    fast_png: bool,
) -> Result<(), Box<dyn Error>> {
    let len = res.len() as u32;
    let w = match width {
        Some(x) => {
            // Width must be at least `k` pixels wide
            if x < len {
                len
            } else {
                x
            }
        }
        None => height * len,
    };

    let mut imgbuf: image::RgbImage = image::ImageBuffer::new(w, height);

    if !proportional {
        let colors: Vec<[u8; 3]> = res
            .iter()
            .map(|r| r.centroid.into_color().into_format().into())
            .collect();
        for (x, _, pixel) in imgbuf.enumerate_pixels_mut() {
            let color = colors
                .get(
                    (((x as f32 / w as f32) * len as f32 - 0.5)
                        .max(0.0)
                        .min(len as f32))
                    .round() as usize,
                )
                .unwrap();
            *pixel = image::Rgb(*color);
        }
    } else {
        let mut curr_pos = 0;
        if let Some((last, elements)) = res.split_last() {
            for r in elements.iter() {
                let pix: [u8; 3] = r.centroid.into_color().into_format().into();
                // Clamp boundary to image width
                let boundary =
                    ((curr_pos as f32 + (r.percentage * w as f32)).round() as u32).min(w);
                for y in 0..height {
                    for x in curr_pos..boundary {
                        imgbuf.put_pixel(x, y, image::Rgb(pix));
                    }
                }
                // If boundary has been clamped, return early
                if boundary == w {
                    return save_image(imgbuf.as_raw(), w, height, title, true, fast_png);
                }
                curr_pos = boundary;
            }
            let pix: [u8; 3] = last.centroid.into_color().into_format().into();
            for y in 0..height {
                for x in curr_pos..w {
                    imgbuf.put_pixel(x, y, image::Rgb(pix));
                }
            }
        }
    }

    save_image(imgbuf.as_raw(), w, height, title, true, fast_png)
}

/// Map the assignments of fully opaque pixels back into the original image.
pub fn map_opaque_pixels(
    original: &[Srgba<u8>],
    centroids: &[Srgba<u8>],
    indices: &[u8],
) -> Vec<Srgba<u8>> {
    let mut indices = indices.iter();
    original
        .iter()
        .map(|pixel| {
            if pixel.alpha == 255 {
                let index = *indices.next().unwrap() as usize;
                *centroids
                    .get(index)
                    .unwrap_or_else(|| centroids.last().unwrap())
            } else {
                Srgba::new(0, 0, 0, 0)
            }
        })
        .collect()
}

/// Restore cluster-index order, using the last replacement for unused clusters.
pub fn indexed_palette<C: Calculate + Copy>(colors: &[CentroidData<C>], len: usize) -> Vec<C> {
    let mut palette = vec![colors.last().unwrap().centroid; len];
    for color in colors {
        palette[color.index as usize] = color.centroid;
    }
    palette
}

/// Optimized conversion of colors from Srgb to Lab using a hashmap for caching
/// of expensive color conversions.
///
/// Additionally, converting from Srgb to Linear Srgb is special-cased in
/// `palette` to use a lookup table which is faster than the regular conversion
/// using `color.into_format().into_color()`.
pub fn cached_srgba_to_lab<'a>(
    rgb: impl Iterator<Item = &'a Srgba<u8>>,
    map: &mut hashbrown::HashMap<[u8; 3], Lab<D65, f32>>,
    lab_pixels: &mut Vec<Lab<D65, f32>>,
) {
    lab_pixels.extend(rgb.map(|color| {
        *map.entry([color.red, color.green, color.blue])
            .or_insert_with(|| color.into_linear::<_, f32>().into_color())
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opaque_assignments_skip_transparent_pixels() {
        let transparent = Srgba::new(0, 0, 0, 0);
        let red = Srgba::new(255, 0, 0, 255);
        let green = Srgba::new(0, 255, 0, 255);
        let original = [
            Srgba::new(1, 2, 3, 0),
            red,
            Srgba::new(1, 2, 3, 254),
            green,
            transparent,
        ];
        assert_eq!(
            map_opaque_pixels(&original, &[red, green], &[1, 0]),
            [transparent, green, transparent, red, transparent],
        );
        assert_eq!(map_opaque_pixels(&[transparent], &[], &[]), [transparent]);
    }

    #[test]
    fn replacement_palette_preserves_indices_of_nonempty_clusters() {
        let red = Srgb::new(1.0, 0.0, 0.0);
        let green = Srgb::new(0.0, 1.0, 0.0);
        let colors = [
            CentroidData {
                centroid: red,
                percentage: 0.75,
                index: 3,
            },
            CentroidData {
                centroid: green,
                percentage: 0.25,
                index: 1,
            },
        ];
        assert_eq!(indexed_palette(&colors, 4), [green, green, green, red]);
    }
}
