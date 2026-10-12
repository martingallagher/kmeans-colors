#![cfg(feature = "app")]

use std::process::Command;

#[test]
fn diverse_cli_preserves_accents_percentages_and_transparency() {
    let dir = std::env::temp_dir().join(format!("kmeans-diverse-cli-{}", std::process::id()));
    let mut pixels = [0, 0, 255, 255].repeat(99);

    pixels.extend_from_slice(&[255, 0, 0, 255, 0, 255, 0, 127, 255, 255, 255, 0]);
    std::fs::create_dir_all(&dir).unwrap();
    image::save_buffer(
        dir.join("input.png"),
        &pixels,
        102,
        1,
        image::ColorType::Rgba8,
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_kmeans_colors"))
        .current_dir(&dir)
        .args([
            "-i",
            "input.png",
            "--diverse",
            "-k",
            "2",
            "--transparent",
            "--pct",
            "--sort",
            "--palette",
            "--proportional",
            "--fast-png",
            "--output",
            "output.png",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        "0000ff,ff0000\n0.9900,0.0100\n"
    );

    let actual = image::open(dir.join("output.png")).unwrap().into_rgba8();
    let palette_path = std::fs::read_dir(&dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .find(|path| path.to_string_lossy().ends_with("-diverse-2.png"))
        .unwrap();
    let palette = image::open(palette_path).unwrap().into_rgb8();

    pixels[400..].fill(0);
    assert_eq!(actual.into_raw(), pixels);
    assert!(palette.pixels().any(|pixel| pixel.0 == [0, 0, 255]));
    assert!(palette.pixels().any(|pixel| pixel.0 == [255, 0, 0]));

    let output = Command::new(env!("CARGO_BIN_EXE_kmeans_colors"))
        .current_dir(&dir)
        .args([
            "-i",
            "input.png",
            "--diverse",
            "--oklab",
            "-k",
            "2",
            "--transparent",
            "--pct",
            "--no-file",
            "--output",
            "unused.png",
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        "0000ff,ff0000\n0.9900,0.0100\n"
    );
    assert!(!dir.join("unused.png").exists());

    let output = Command::new(env!("CARGO_BIN_EXE_kmeans_colors"))
        .current_dir(&dir)
        .args([
            "-i",
            "input.png",
            "--diverse",
            "-k",
            "4",
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

    let opaque = image::open(dir.join("opaque.png")).unwrap().into_rgb8();
    let mut expected = [0, 0, 255].repeat(99);

    expected.extend_from_slice(&[255, 0, 0, 0, 255, 0, 255, 255, 255]);
    assert_eq!(opaque.into_raw(), expected);

    for args in [
        vec!["--rgb"],
        vec!["--iterations", "5"],
        vec!["--factor", "0.0001"],
        vec!["--runs", "2"],
        vec!["--seed", "42"],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_kmeans_colors"))
            .current_dir(&dir)
            .args(["-i", "input.png", "--diverse", "--no-file"])
            .args(&args)
            .output()
            .unwrap();

        assert!(
            !output.status.success(),
            "accepted incompatible arguments: {:?}",
            args
        );
        assert!(String::from_utf8_lossy(&output.stderr).contains("cannot be used with"));
    }

    image::save_buffer(
        dir.join("empty.png"),
        &[0; 4],
        1,
        1,
        image::ColorType::Rgba8,
    )
    .unwrap();

    for (input, k, message) in [
        ("input.png", "0", "requires at least one color"),
        ("empty.png", "2", "No pixels available"),
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_kmeans_colors"))
            .current_dir(&dir)
            .args([
                "-i",
                input,
                "--diverse",
                "--transparent",
                "-k",
                k,
                "--output",
                "invalid.png",
            ])
            .output()
            .unwrap();

        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains(message));
        assert!(!dir.join("invalid.png").exists());
    }

    std::fs::remove_dir_all(dir).unwrap();
}
