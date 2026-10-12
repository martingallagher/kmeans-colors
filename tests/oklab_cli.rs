#![cfg(feature = "app")]

use std::process::Command;

#[test]
fn cli_color_spaces_convert_images_palettes_and_replacements() {
    let dir = std::env::temp_dir().join(format!("kmeans-oklab-cli-{}", std::process::id()));

    std::fs::create_dir_all(&dir).unwrap();
    image::save_buffer(
        dir.join("input.png"),
        &[
            0, 0, 0, 255, 255, 255, 255, 255, 255, 0, 0, 127, 0, 0, 255, 0,
        ],
        4,
        1,
        image::ColorType::Rgba8,
    )
    .unwrap();

    for (space, gray) in [(None, 119), (Some("--rgb"), 128), (Some("--oklab"), 99)] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_kmeans_colors"));

        command.current_dir(&dir).args([
            "-i",
            "input.png",
            "-k",
            "1",
            "--transparent",
            "--seed",
            "42",
            "--runs",
            "2",
            "--pct",
            "--palette",
            "--op",
            "palette.png",
            "--output",
            "output.png",
        ]);

        if let Some(space) = space {
            command.arg(space);
        }

        let output = command.output().unwrap();

        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            String::from_utf8(output.stdout).unwrap(),
            format!("{gray:02x}{gray:02x}{gray:02x}\n1.0000\n")
        );

        let actual = image::open(dir.join("output.png")).unwrap().into_rgba8();
        let palette = image::open(dir.join("palette.png")).unwrap().into_rgb8();

        assert_eq!(
            actual.into_raw(),
            [gray, gray, gray, 255, gray, gray, gray, 255, 0, 0, 0, 0, 0, 0, 0, 0]
        );
        assert!(palette.pixels().all(|p| p.0 == [gray; 3]));
    }

    let output = Command::new(env!("CARGO_BIN_EXE_kmeans_colors"))
        .current_dir(&dir)
        .args([
            "-i",
            "input.png",
            "-k",
            "4",
            "--oklab",
            "--seed",
            "42",
            "--runs",
            "2",
            "--pct",
            "--palette",
            "--proportional",
            "--sort",
            "--output",
            "opaque.png",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8(output.stdout).unwrap();
    let mut colors: Vec<_> = stdout.lines().next().unwrap().split(',').collect();

    colors.sort_unstable();
    assert_eq!(colors, ["000000", "0000ff", "ff0000", "ffffff"]);
    assert_eq!(stdout.lines().nth(1), Some("0.2500,0.2500,0.2500,0.2500"));
    assert_eq!(
        image::open(dir.join("opaque.png"))
            .unwrap()
            .into_rgb8()
            .into_raw(),
        [0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 255]
    );
    assert!(std::fs::read_dir(&dir).unwrap().any(|entry| entry
        .unwrap()
        .file_name()
        .to_string_lossy()
        .ends_with("-oklab-4.png")));

    for replace in [false, true] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_kmeans_colors"));

        command.current_dir(&dir).args([
            "find",
            "-i",
            "input.png",
            "--oklab",
            "--transparent",
            "--seed",
            "42",
            "--colors",
            "ff0000,00ff00",
            "--output",
            "find.png",
            "--pct",
        ]);

        if replace {
            command.arg("--replace");
        }

        let output = command.output().unwrap();

        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );

        let actual = image::open(dir.join("find.png")).unwrap().into_rgba8();

        assert_eq!(
            actual.into_raw(),
            [255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0]
        );
    }

    for args in [
        vec!["--rgb", "--oklab"],
        vec![
            "find",
            "-i",
            "input.png",
            "-c",
            "000000,ffffff",
            "--rgb",
            "--oklab",
        ],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_kmeans_colors"))
            .current_dir(&dir)
            .args(args)
            .output()
            .unwrap();

        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("cannot be used with"));
    }

    std::fs::remove_dir_all(dir).unwrap();
}
