use spritesheet_detector::{SpritesheetInfo, analyze_spritesheet};

#[test]
fn test_regression_archer_card_icon() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/regression_archer_card_icon.png").unwrap(),),
        SpritesheetInfo {
            sprite_width: 500,
            sprite_height: 500,
            columns: 1,
            rows: 1,
            frame_count: 1
        }
    );
}

#[test]
fn test_regression_castle_card_human() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/regression_castle_card_human.png").unwrap(),),
        SpritesheetInfo {
            sprite_width: 220,
            sprite_height: 388,
            columns: 1,
            rows: 1,
            frame_count: 1
        }
    );
}

#[test]
fn test_regression_bomb_card_projectile() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/regression_bomb_card_projectile.png").unwrap(),),
        SpritesheetInfo {
            sprite_width: 293,
            sprite_height: 252,
            columns: 1,
            rows: 1,
            frame_count: 1
        }
    );
}

#[test]
fn test_regression_archer_card_blue_idle_n() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/regression_archer_card_blue_idle_n.png").unwrap()),
        SpritesheetInfo {
            sprite_width: 400,
            sprite_height: 400,
            columns: 1,
            rows: 1,
            frame_count: 1
        }
    );
}

#[test]
fn test_regression_bomber_card_blue_n() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/regression_bomber_card_blue_n.png").unwrap()),
        SpritesheetInfo {
            sprite_width: 223,
            sprite_height: 416,
            columns: 1,
            rows: 1,
            frame_count: 1
        }
    );
}

#[test]
fn test_regression_nav_main() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/regression_nav_main.png").unwrap()),
        SpritesheetInfo {
            sprite_width: 250,
            sprite_height: 250,
            columns: 1,
            rows: 1,
            frame_count: 1
        }
    );
}

#[test]
fn test_regression_coffin_maw_card() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/regression_coffin_maw_card.png").unwrap()),
        SpritesheetInfo {
            sprite_width: 400,
            sprite_height: 400,
            columns: 1,
            rows: 1,
            frame_count: 1
        }
    );
}

#[test]
fn test_regression_crossbow_card() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/regression_crossbow_card.png").unwrap()),
        SpritesheetInfo {
            sprite_width: 383,
            sprite_height: 320,
            columns: 1,
            rows: 1,
            frame_count: 1
        }
    );
}

#[test]
fn test_regression_border_s_1() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/regression_border_s_1.png").unwrap()),
        SpritesheetInfo {
            sprite_width: 200,
            sprite_height: 1008,
            columns: 1,
            rows: 1,
            frame_count: 1
        }
    );
}

#[test]
fn test_regression_cardframes_gold() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/regression_cardframes_gold.png").unwrap()),
        SpritesheetInfo {
            sprite_width: 720,
            sprite_height: 1000,
            columns: 1,
            rows: 1,
            frame_count: 1
        }
    );
}

#[test]
fn test_regression_cardframes_ranged() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/regression_cardframes_ranged.png").unwrap()),
        SpritesheetInfo {
            sprite_width: 720,
            sprite_height: 1000,
            columns: 1,
            rows: 1,
            frame_count: 1
        }
    );
}

#[test]
fn test_regression_cardframes_tower() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/regression_cardframes_tower.png").unwrap()),
        SpritesheetInfo {
            sprite_width: 720,
            sprite_height: 1000,
            columns: 1,
            rows: 1,
            frame_count: 1
        }
    );
}

#[test]
fn test_asset_example() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/example.png").unwrap()),
        SpritesheetInfo {
            sprite_width: 193,
            sprite_height: 155,
            columns: 5,
            rows: 4,
            frame_count: 18,
        }
    );
}

#[test]
fn test_map_tiles() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/map_tiles.png").unwrap()),
        SpritesheetInfo {
            sprite_width: 200,
            sprite_height: 168,
            columns: 2,
            rows: 3,
            frame_count: 6,
        }
    );
}

#[test]
#[ignore = "too complicated for now"]
fn test_map_overlays() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/map_overlays.png").unwrap()),
        SpritesheetInfo {
            sprite_width: 200,
            sprite_height: 168,
            columns: 9,
            rows: 3,
            frame_count: 27,
        }
    );
}

#[test]
fn test_map_tiles_borders() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/map_tiles_borders.png").unwrap()),
        SpritesheetInfo {
            sprite_width: 400,
            sprite_height: 336,
            columns: 8,
            rows: 3,
            frame_count: 20,
        }
    );
}

#[test]
fn test_bomb_card_area() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/bomb_card_area.png").unwrap()),
        SpritesheetInfo {
            sprite_width: 449,
            sprite_height: 318,
            columns: 4,
            rows: 3,
            frame_count: 10,
        }
    );
}

#[test]
fn test_lightning_mage_card_overlay() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/lightning_mage_card_overlay.png").unwrap(),),
        SpritesheetInfo {
            sprite_width: 266,
            sprite_height: 327,
            columns: 9,
            rows: 8,
            frame_count: 72,
        }
    );
}

#[test]
fn test_cannon_card_projectile_impact_small() {
    assert_eq!(
        analyze_spritesheet(
            &image::open("assets/cannon_card_projectile_impact_small.png").unwrap(),
        ),
        SpritesheetInfo {
            sprite_width: 101,
            sprite_height: 102,
            columns: 4,
            rows: 3,
            frame_count: 10,
        }
    );
}

#[test]
fn test_ghoul_ripper_troop_hit() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/ghoul_ripper_troop_hit.png").unwrap(),),
        SpritesheetInfo {
            sprite_width: 136,
            sprite_height: 122,
            columns: 4,
            rows: 3,
            frame_count: 12,
        }
    );
}

#[test]
fn test_totem_tower_totem_troop_vfx() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/totem_tower_totem_troop_vfx.png").unwrap(),),
        SpritesheetInfo {
            sprite_width: 128,
            sprite_height: 84,
            columns: 5,
            rows: 4,
            frame_count: 19,
        }
    );
}

#[test]
fn test_totem_tower_troop_projectile_impact() {
    assert_eq!(
        analyze_spritesheet(
            &image::open("assets/totem_tower_troop_projectile_impact.png").unwrap(),
        ),
        SpritesheetInfo {
            sprite_width: 97,
            sprite_height: 81,
            columns: 4,
            rows: 3,
            frame_count: 10,
        }
    );
}

#[test]
fn test_totem_tower_totem_troop_active() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/totem_tower_totem_troop_active.png").unwrap(),),
        SpritesheetInfo {
            sprite_width: 96,
            sprite_height: 188,
            columns: 5,
            rows: 4,
            frame_count: 20,
        }
    );
}

#[test]
fn test_totem_tower_totem_troop_spawn() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/totem_tower_totem_troop_spawn.png").unwrap(),),
        SpritesheetInfo {
            sprite_width: 96,
            sprite_height: 188,
            columns: 4,
            rows: 3,
            frame_count: 10,
        }
    );
}

#[test]
fn test_totem_tower_totem_troop_death() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/totem_tower_totem_troop_death.png").unwrap(),),
        SpritesheetInfo {
            sprite_width: 96,
            sprite_height: 188,
            columns: 3,
            rows: 3,
            frame_count: 8,
        }
    );
}

#[test]
fn test_building_construction() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/building_construction.png").unwrap(),),
        SpritesheetInfo {
            sprite_width: 248,
            sprite_height: 322,
            columns: 7,
            rows: 7,
            frame_count: 48,
        }
    );
}

#[test]
fn test_building_deconstruction() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/building_deconstruction.png").unwrap(),),
        SpritesheetInfo {
            sprite_width: 248,
            sprite_height: 322,
            columns: 8,
            rows: 7,
            frame_count: 55,
        }
    );
}

#[test]
fn test_building_destruction() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/building_destruction.png").unwrap()),
        SpritesheetInfo {
            sprite_width: 340,
            sprite_height: 322,
            columns: 8,
            rows: 7,
            frame_count: 53,
        }
    );
}

#[test]
fn test_peasant_card_overlay() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/peasant_card_overlay.png").unwrap()),
        SpritesheetInfo {
            sprite_width: 383,
            sprite_height: 392,
            columns: 11,
            rows: 10,
            frame_count: 100,
        }
    );
}

#[test]
fn test_necromancer_troop_projectile() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/necromancer_troop_projectile.png").unwrap(),),
        SpritesheetInfo {
            sprite_width: 77,
            sprite_height: 49,
            columns: 3,
            rows: 3,
            frame_count: 8,
        }
    );
}

#[test]
fn test_necromancer_troop_projectile_impact() {
    assert_eq!(
        analyze_spritesheet(
            &image::open("assets/necromancer_troop_projectile_impact.png").unwrap(),
        ),
        SpritesheetInfo {
            sprite_width: 82,
            sprite_height: 74,
            columns: 3,
            rows: 3,
            frame_count: 8,
        }
    );
}

#[test]
fn test_bomber_troop_projectile() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/bomber_troop_projectile.png").unwrap(),),
        SpritesheetInfo {
            sprite_width: 73,
            sprite_height: 71,
            columns: 2,
            rows: 2,
            frame_count: 4,
        }
    );
}

#[test]
fn test_monk_tower_troop_rock_attack() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/monk_tower_troop_rock_attack.png").unwrap(),),
        SpritesheetInfo {
            sprite_width: 238,
            sprite_height: 706,
            columns: 6,
            rows: 6,
            frame_count: 35,
        }
    );
}

#[test]
fn test_chest_notif() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/chest_notif.png").unwrap()),
        SpritesheetInfo {
            sprite_width: 99,
            sprite_height: 117,
            columns: 3,
            rows: 3,
            frame_count: 7,
        }
    );
}

#[test]
fn test_archer_troop_blue_projectile() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/archer_troop_blue_projectile.png").unwrap(),),
        SpritesheetInfo {
            sprite_width: 106,
            sprite_height: 15,
            columns: 1,
            rows: 1,
            frame_count: 1
        }
    );
}

#[test]
fn test_mask_troop_projectile() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/mask_troop_projectile.png").unwrap(),),
        SpritesheetInfo {
            sprite_width: 108,
            sprite_height: 62,
            columns: 1,
            rows: 1,
            frame_count: 1
        }
    );
}

#[test]
fn test_ghoul_ripper_card_blue_peek() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/ghoul_ripper_card_blue_peek.png").unwrap(),),
        SpritesheetInfo {
            sprite_width: 320,
            sprite_height: 320,
            columns: 7,
            rows: 6,
            frame_count: 38
        }
    );
}

#[test]
fn test_regression_cardframes_pricetag() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/regression_cardframes_pricetag.png").unwrap()),
        SpritesheetInfo {
            sprite_width: 720,
            sprite_height: 1000,
            columns: 1,
            rows: 1,
            frame_count: 1
        }
    );
}

#[test]
fn test_regression_chest_common_open_idle() {
    let result =
        analyze_spritesheet(&image::open("assets/regression_chest_common_open_idle.png").unwrap());
    assert_eq!(result.columns, 8);
    assert_eq!(result.rows, 8);
}

#[test]
fn test_regression_totem_tower_idle() {
    let result =
        analyze_spritesheet(&image::open("assets/regression_totem_tower_idle.png").unwrap());
    assert_eq!(result.columns, 8);
    assert_eq!(result.rows, 7);
}

#[test]
fn test_regression_priest_heal() {
    let result = analyze_spritesheet(&image::open("assets/regression_priest_heal.png").unwrap());
    assert_eq!(result.columns, 9);
    assert_eq!(result.rows, 9);
}

#[test]
fn test_regression_hero_builder_idle() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/regression_hero_builder_idle.png").unwrap()),
        SpritesheetInfo {
            sprite_width: 100,
            sprite_height: 64,
            columns: 1,
            rows: 1,
            frame_count: 1
        }
    );
}

#[test]
fn test_regression_crossbow_projectile() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/regression_crossbow_projectile.png").unwrap()),
        SpritesheetInfo {
            sprite_width: 73,
            sprite_height: 28,
            columns: 1,
            rows: 1,
            frame_count: 1
        }
    );
}

#[test]
#[ignore]
fn test_ghoul_ripper_troop_blue_spawn() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/ghoul_ripper_troop_blue_spawn.png").unwrap(),),
        SpritesheetInfo {
            sprite_width: 280,
            sprite_height: 224,
            columns: 6,
            rows: 5,
            frame_count: 29
        }
    );
}

#[test]
fn test_hero_builder_troop_blue_attack() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/hero_builder_troop_blue_attack.png").unwrap(),),
        SpritesheetInfo {
            sprite_width: 100,
            sprite_height: 64,
            columns: 4,
            rows: 3,
            frame_count: 11
        }
    );
}

#[test]
#[ignore]
fn test_punk_band_guitarrist_troop_blue_death() {
    assert_eq!(
        analyze_spritesheet(
            &image::open("assets/punk_band_guitarrist_troop_blue_death.png").unwrap(),
        ),
        SpritesheetInfo {
            sprite_width: 340,
            sprite_height: 304,
            columns: 4,
            rows: 5,
            frame_count: 20
        }
    );
}

#[test]
fn test_ui_chest_lock() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/UI_chest_lock.png").unwrap()),
        SpritesheetInfo {
            sprite_width: 130,
            sprite_height: 130,
            columns: 3,
            rows: 2,
            frame_count: 6
        }
    );
}

#[test]
fn test_card_glow() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/card_glow.png").unwrap()),
        SpritesheetInfo {
            sprite_width: 438,
            sprite_height: 514,
            columns: 2,
            rows: 2,
            frame_count: 4
        }
    );
}

#[test]
fn test_blue_cross() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/blue_cross.png").unwrap()),
        SpritesheetInfo {
            sprite_width: 200,
            sprite_height: 200,
            columns: 1,
            rows: 1,
            frame_count: 1
        }
    );
}
