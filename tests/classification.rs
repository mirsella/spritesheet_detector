use image::{DynamicImage, GenericImageView, Rgba, RgbaImage};
use spritesheet_detector::{SpritesheetInfo, analyze_spritesheet};

fn grid(info: &SpritesheetInfo) -> (u32, u32, u32, u32, u32) {
    (
        info.sprite_width,
        info.sprite_height,
        info.columns,
        info.rows,
        info.frame_count,
    )
}

#[test]
fn single_images_with_internal_edges() {
    for name in [
        "settings_chat.png",
        "settings_network-4.png",
        "settings_screen.png",
        "settings_target.png",
        "settings_volume-up.png",
        "feedback_bug.png",
        "feedback_voice_resume.png",
        "pause.png",
        "restart.png",
        "envelope.png",
        "voxride-wordmark.png",
        "card_compare.png",
        "card_CommonFrame.png",
        "card_CommonInnerFrame.png",
        "card_UncommonInnerFrame.png",
        "card_RareInnerFrame.png",
        "card_LegendaryInnerFrame.png",
        "map_dark_forest_border_s_1.png",
        "map_snow_territory_blue_edge_s.png",
        "map_snow_territory_red_edge_s.png",
    ] {
        let image = image::open(format!("assets/classification/{name}")).unwrap();
        let (width, height) = image.dimensions();
        assert_eq!(
            grid(&analyze_spritesheet(&image)),
            (width, height, 1, 1, 1),
            "{name}"
        );
    }
}

#[test]
fn source_animations_with_changing_or_sparse_frames() {
    for (name, expected) in [
        ("source_zap_card_hit.png", (100, 103, 4, 4, 13)),
        ("lightning_mage_troop_projectile.png", (168, 96, 5, 4, 17)),
        ("musketeer_troop_projectile_impact.png", (90, 89, 2, 2, 4)),
        ("cursor_click_n.png", (500, 580, 2, 1, 2)),
    ] {
        let image = image::open(format!("assets/classification/{name}")).unwrap();
        assert_eq!(grid(&analyze_spritesheet(&image)), expected, "{name}");
    }
}

#[test]
fn repeated_ui_animation_keeps_dissimilar_rows_and_columns() {
    let image = image::open("assets/UI_chest_lock.png").unwrap();
    assert_eq!(grid(&analyze_spritesheet(&image)), (130, 130, 3, 2, 6));
    assert_eq!(
        grid(&analyze_spritesheet(&image.rotate90())),
        (130, 130, 2, 3, 6)
    );
}

fn replace_invisible_rgb(image: &mut RgbaImage) {
    for (x, y, pixel) in image.enumerate_pixels_mut() {
        if pixel[3] == 0 {
            let value = if (x / 16 + y / 16) % 2 == 0 { 255 } else { 0 };
            *pixel = Rgba([value, value, value, 0]);
        }
    }
}

#[test]
fn invisible_rgb_cannot_create_frames() {
    let mut rgba = RgbaImage::new(256, 256);
    replace_invisible_rgb(&mut rgba);
    assert_eq!(
        grid(&analyze_spritesheet(&DynamicImage::ImageRgba8(rgba))),
        (256, 256, 1, 1, 0)
    );
}

#[test]
fn invisible_rgb_does_not_change_static_or_animation_detection() {
    for path in [
        "assets/classification/settings_chat.png",
        "assets/classification/lightning_mage_troop_projectile.png",
        "assets/classification/source_zap_card_hit.png",
        "assets/UI_chest_lock.png",
        "assets/card_glow.png",
    ] {
        let image = image::open(path).unwrap();
        let expected = grid(&analyze_spritesheet(&image));
        let mut rgba = image.to_rgba8();
        replace_invisible_rgb(&mut rgba);
        assert_eq!(
            grid(&analyze_spritesheet(&DynamicImage::ImageRgba8(rgba))),
            expected,
            "{path}"
        );
    }
}
