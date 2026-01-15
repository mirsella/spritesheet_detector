use spritesheet_detector::{SpritesheetInfo, analyze_spritesheet};

fn test_asset(path: &str, expected: SpritesheetInfo) {
    let img = image::open(path).expect(&format!("Failed to open {}", path));
    assert_eq!(analyze_spritesheet(&img), expected, "Mismatch for {}", path);
}

#[test]
fn test_archer_card_icon() {
    test_asset(
        "assets/regression_archer_card_icon.png",
        SpritesheetInfo {
            sprite_width: 500,
            sprite_height: 500,
            columns: 1,
            rows: 1,
            frame_count: 1,
        },
    );
}

#[test]
fn test_castle_card_human() {
    test_asset(
        "assets/regression_castle_card_human.png",
        SpritesheetInfo {
            sprite_width: 220,
            sprite_height: 388,
            columns: 1,
            rows: 1,
            frame_count: 1,
        },
    );
}

#[test]
fn test_bomb_card_projectile() {
    test_asset(
        "assets/regression_bomb_card_projectile.png",
        SpritesheetInfo {
            sprite_width: 293,
            sprite_height: 252,
            columns: 1,
            rows: 1,
            frame_count: 1,
        },
    );
}

#[test]
fn test_archer_card_blue_idle_n() {
    test_asset(
        "assets/regression_archer_card_blue_idle_n.png",
        SpritesheetInfo {
            sprite_width: 400,
            sprite_height: 400,
            columns: 1,
            rows: 1,
            frame_count: 1,
        },
    );
}

#[test]
fn test_bomber_card_blue_n() {
    test_asset(
        "assets/regression_bomber_card_blue_n.png",
        SpritesheetInfo {
            sprite_width: 223,
            sprite_height: 416,
            columns: 1,
            rows: 1,
            frame_count: 1,
        },
    );
}

#[test]
fn test_nav_main() {
    test_asset(
        "assets/regression_nav_main.png",
        SpritesheetInfo {
            sprite_width: 250,
            sprite_height: 250,
            columns: 1,
            rows: 1,
            frame_count: 1,
        },
    );
}

#[test]
fn test_coffin_maw_card() {
    test_asset(
        "assets/regression_coffin_maw_card.png",
        SpritesheetInfo {
            sprite_width: 400,
            sprite_height: 400,
            columns: 1,
            rows: 1,
            frame_count: 1,
        },
    );
}

#[test]
fn test_crossbow_card() {
    test_asset(
        "assets/regression_crossbow_card.png",
        SpritesheetInfo {
            sprite_width: 383,
            sprite_height: 320,
            columns: 1,
            rows: 1,
            frame_count: 1,
        },
    );
}

#[test]
fn test_border_s_1() {
    test_asset(
        "assets/regression_border_s_1.png",
        SpritesheetInfo {
            sprite_width: 200,
            sprite_height: 1008,
            columns: 1,
            rows: 1,
            frame_count: 1,
        },
    );
}

#[test]
fn test_cardframes_gold() {
    test_asset(
        "assets/regression_cardframes_gold.png",
        SpritesheetInfo {
            sprite_width: 720,
            sprite_height: 1000,
            columns: 1,
            rows: 1,
            frame_count: 1,
        },
    );
}

#[test]
fn test_cardframes_ranged() {
    test_asset(
        "assets/regression_cardframes_ranged.png",
        SpritesheetInfo {
            sprite_width: 720,
            sprite_height: 1000,
            columns: 1,
            rows: 1,
            frame_count: 1,
        },
    );
}

#[test]
fn test_cardframes_tower() {
    test_asset(
        "assets/regression_cardframes_tower.png",
        SpritesheetInfo {
            sprite_width: 720,
            sprite_height: 1000,
            columns: 1,
            rows: 1,
            frame_count: 1,
        },
    );
}

#[test]
fn test_asset_example() {
    test_asset(
        "assets/example.png",
        SpritesheetInfo {
            sprite_width: 193,
            sprite_height: 155,
            columns: 5,
            rows: 4,
            frame_count: 18,
        },
    );
}

#[test]
fn test_map_tiles() {
    test_asset(
        "assets/map_tiles.png",
        SpritesheetInfo {
            sprite_width: 200,
            sprite_height: 168,
            columns: 2,
            rows: 3,
            frame_count: 6,
        },
    );
}

#[test]
#[ignore = "too complicated for now"]
fn test_map_overlays() {
    test_asset(
        "assets/map_overlays.png",
        SpritesheetInfo {
            sprite_width: 200,
            sprite_height: 168,
            columns: 9,
            rows: 3,
            frame_count: 27,
        },
    );
}

#[test]
fn test_map_tiles_borders() {
    test_asset(
        "assets/map_tiles_borders.png",
        SpritesheetInfo {
            sprite_width: 400,
            sprite_height: 336,
            columns: 8,
            rows: 3,
            frame_count: 20,
        },
    );
}

#[test]
fn test_bomb_card_area() {
    test_asset(
        "assets/bomb_card_area.png",
        SpritesheetInfo {
            sprite_width: 449,
            sprite_height: 318,
            columns: 4,
            rows: 3,
            frame_count: 10,
        },
    );
}

#[test]
fn test_lightning_mage_card_overlay() {
    test_asset(
        "assets/lightning_mage_card_overlay.png",
        SpritesheetInfo {
            sprite_width: 266,
            sprite_height: 327,
            columns: 9,
            rows: 8,
            frame_count: 72,
        },
    );
}

#[test]
fn test_cannon_card_projectile_impact_small() {
    test_asset(
        "assets/cannon_card_projectile_impact_small.png",
        SpritesheetInfo {
            sprite_width: 101,
            sprite_height: 102,
            columns: 4,
            rows: 3,
            frame_count: 10,
        },
    );
}

#[test]
fn test_ghoul_ripper_troop_hit() {
    test_asset(
        "assets/ghoul_ripper_troop_hit.png",
        SpritesheetInfo {
            sprite_width: 136,
            sprite_height: 122,
            columns: 4,
            rows: 3,
            frame_count: 12,
        },
    );
}

#[test]
fn test_totem_tower_totem_troop_vfx() {
    test_asset(
        "assets/totem_tower_totem_troop_vfx.png",
        SpritesheetInfo {
            sprite_width: 128,
            sprite_height: 84,
            columns: 5,
            rows: 4,
            frame_count: 19,
        },
    );
}

#[test]
fn test_totem_tower_troop_projectile_impact() {
    test_asset(
        "assets/totem_tower_troop_projectile_impact.png",
        SpritesheetInfo {
            sprite_width: 97,
            sprite_height: 81,
            columns: 4,
            rows: 3,
            frame_count: 10,
        },
    );
}

#[test]
fn test_totem_tower_totem_troop_active() {
    test_asset(
        "assets/totem_tower_totem_troop_active.png",
        SpritesheetInfo {
            sprite_width: 96,
            sprite_height: 188,
            columns: 5,
            rows: 4,
            frame_count: 20,
        },
    );
}

#[test]
fn test_totem_tower_totem_troop_spawn() {
    test_asset(
        "assets/totem_tower_totem_troop_spawn.png",
        SpritesheetInfo {
            sprite_width: 96,
            sprite_height: 188,
            columns: 4,
            rows: 3,
            frame_count: 10,
        },
    );
}

#[test]
fn test_totem_tower_totem_troop_death() {
    test_asset(
        "assets/totem_tower_totem_troop_death.png",
        SpritesheetInfo {
            sprite_width: 96,
            sprite_height: 188,
            columns: 3,
            rows: 3,
            frame_count: 8,
        },
    );
}

#[test]
fn test_building_construction() {
    test_asset(
        "assets/building_construction.png",
        SpritesheetInfo {
            sprite_width: 248,
            sprite_height: 322,
            columns: 7,
            rows: 7,
            frame_count: 48,
        },
    );
}

#[test]
fn test_building_deconstruction() {
    test_asset(
        "assets/building_deconstruction.png",
        SpritesheetInfo {
            sprite_width: 248,
            sprite_height: 322,
            columns: 8,
            rows: 7,
            frame_count: 55,
        },
    );
}

#[test]
fn test_building_destruction() {
    test_asset(
        "assets/building_destruction.png",
        SpritesheetInfo {
            sprite_width: 340,
            sprite_height: 322,
            columns: 8,
            rows: 7,
            frame_count: 53,
        },
    );
}

#[test]
fn test_peasant_card_overlay() {
    test_asset(
        "assets/peasant_card_overlay.png",
        SpritesheetInfo {
            sprite_width: 383,
            sprite_height: 392,
            columns: 11,
            rows: 10,
            frame_count: 100,
        },
    );
}

#[test]
fn test_necromancer_troop_projectile() {
    test_asset(
        "assets/necromancer_troop_projectile.png",
        SpritesheetInfo {
            sprite_width: 77,
            sprite_height: 49,
            columns: 3,
            rows: 3,
            frame_count: 8,
        },
    );
}

#[test]
fn test_necromancer_troop_projectile_impact() {
    test_asset(
        "assets/necromancer_troop_projectile_impact.png",
        SpritesheetInfo {
            sprite_width: 82,
            sprite_height: 74,
            columns: 3,
            rows: 3,
            frame_count: 8,
        },
    );
}

#[test]
fn test_bomber_troop_projectile() {
    test_asset(
        "assets/bomber_troop_projectile.png",
        SpritesheetInfo {
            sprite_width: 73,
            sprite_height: 71,
            columns: 2,
            rows: 2,
            frame_count: 4,
        },
    );
}

#[test]
fn test_monk_tower_troop_rock_attack() {
    test_asset(
        "assets/monk_tower_troop_rock_attack.png",
        SpritesheetInfo {
            sprite_width: 238,
            sprite_height: 706,
            columns: 6,
            rows: 6,
            frame_count: 35,
        },
    );
}

#[test]
fn test_chest_notif() {
    test_asset(
        "assets/chest_notif.png",
        SpritesheetInfo {
            sprite_width: 99,
            sprite_height: 117,
            columns: 3,
            rows: 3,
            frame_count: 7,
        },
    );
}

#[test]
fn test_archer_troop_blue_projectile() {
    test_asset(
        "assets/archer_troop_blue_projectile.png",
        SpritesheetInfo {
            sprite_width: 106,
            sprite_height: 15,
            columns: 1,
            rows: 1,
            frame_count: 1,
        },
    );
}

#[test]
fn test_mask_troop_projectile() {
    test_asset(
        "assets/mask_troop_projectile.png",
        SpritesheetInfo {
            sprite_width: 108,
            sprite_height: 62,
            columns: 1,
            rows: 1,
            frame_count: 1,
        },
    );
}

#[test]
fn test_ghoul_ripper_card_blue_peek() {
    test_asset(
        "assets/ghoul_ripper_card_blue_peek.png",
        SpritesheetInfo {
            sprite_width: 320,
            sprite_height: 320,
            columns: 7,
            rows: 6,
            frame_count: 38,
        },
    );
}

#[test]
fn test_cardframes_pricetag() {
    test_asset(
        "assets/regression_cardframes_pricetag.png",
        SpritesheetInfo {
            sprite_width: 720,
            sprite_height: 1000,
            columns: 1,
            rows: 1,
            frame_count: 1,
        },
    );
}

#[test]
fn test_hero_builder_idle() {
    test_asset(
        "assets/regression_hero_builder_idle.png",
        SpritesheetInfo {
            sprite_width: 100,
            sprite_height: 64,
            columns: 1,
            rows: 1,
            frame_count: 1,
        },
    );
}

#[test]
fn test_crossbow_projectile() {
    test_asset(
        "assets/regression_crossbow_projectile.png",
        SpritesheetInfo {
            sprite_width: 73,
            sprite_height: 28,
            columns: 1,
            rows: 1,
            frame_count: 1,
        },
    );
}

#[test]
fn test_ghoul_ripper_troop_blue_spawn() {
    test_asset(
        "assets/ghoul_ripper_troop_blue_spawn.png",
        SpritesheetInfo {
            sprite_width: 280,
            sprite_height: 224,
            columns: 6,
            rows: 5,
            frame_count: 30,
        },
    );
}

#[test]
fn test_hero_builder_troop_blue_attack() {
    test_asset(
        "assets/hero_builder_troop_blue_attack.png",
        SpritesheetInfo {
            sprite_width: 100,
            sprite_height: 64,
            columns: 4,
            rows: 3,
            frame_count: 11,
        },
    );
}

#[test]
fn test_ui_chest_lock() {
    test_asset(
        "assets/UI_chest_lock.png",
        SpritesheetInfo {
            sprite_width: 130,
            sprite_height: 130,
            columns: 3,
            rows: 2,
            frame_count: 6,
        },
    );
}

#[test]
fn test_card_glow() {
    test_asset(
        "assets/card_glow.png",
        SpritesheetInfo {
            sprite_width: 438,
            sprite_height: 514,
            columns: 2,
            rows: 2,
            frame_count: 4,
        },
    );
}

#[test]
fn test_blue_cross() {
    test_asset(
        "assets/blue_cross.png",
        SpritesheetInfo {
            sprite_width: 200,
            sprite_height: 200,
            columns: 1,
            rows: 1,
            frame_count: 1,
        },
    );
}

#[test]
fn test_archer_troop_projectile_impact() {
    test_asset(
        "assets/archer_troop_projectile_impact.png",
        SpritesheetInfo {
            sprite_width: 90,
            sprite_height: 89,
            columns: 2,
            rows: 2,
            frame_count: 4,
        },
    );
}

#[test]
fn test_punk_band_guitarrist_troop_blue_death() {
    test_asset(
        "assets/punk_band_guitarrist_troop_blue_death.png",
        SpritesheetInfo {
            sprite_width: 340,
            sprite_height: 380,
            columns: 4,
            rows: 4,
            frame_count: 16,
        },
    );
}

#[test]
fn test_punk_band_guitarrist_troop_red_death() {
    test_asset(
        "assets/punk_band_guitarrist_troop_red_death.png",
        SpritesheetInfo {
            sprite_width: 340,
            sprite_height: 380,
            columns: 4,
            rows: 4,
            frame_count: 16,
        },
    );
}

#[test]
fn test_totem_tower_troop_blue_eat_n() {
    test_asset(
        "assets/totem_tower_troop_blue_eat_n.png",
        SpritesheetInfo {
            sprite_width: 296,
            sprite_height: 604,
            columns: 9,
            rows: 8,
            frame_count: 70,
        },
    );
}

#[test]
fn test_priest_troop_blue_heal_unit_n() {
    test_asset(
        "assets/priest_troop_blue_heal_unit_n.png",
        SpritesheetInfo {
            sprite_width: 320,
            sprite_height: 456,
            columns: 9,
            rows: 9,
            frame_count: 78,
        },
    );
}

#[test]
fn test_chest_common_0_lvlup() {
    test_asset(
        "assets/chest_common_0_lvlup.png",
        SpritesheetInfo {
            sprite_width: 942,
            sprite_height: 809,
            columns: 6,
            rows: 6,
            frame_count: 31,
        },
    );
}

#[test]
fn test_chest_common_1_call() {
    test_asset(
        "assets/chest_common_1_call.png",
        SpritesheetInfo {
            sprite_width: 942,
            sprite_height: 809,
            columns: 5,
            rows: 5,
            frame_count: 25,
        },
    );
}

#[test]
fn test_chest_common_1_lvlup() {
    test_asset(
        "assets/chest_common_1_lvlup.png",
        SpritesheetInfo {
            sprite_width: 942,
            sprite_height: 809,
            columns: 6,
            rows: 6,
            frame_count: 31,
        },
    );
}

#[test]
fn test_chest_common_1_open_idle() {
    test_asset(
        "assets/chest_common_1_open_idle.png",
        SpritesheetInfo {
            sprite_width: 942,
            sprite_height: 809,
            columns: 8,
            rows: 8,
            frame_count: 60,
        },
    );
}

#[test]
fn test_totem_tower_troop_blue_idle_n() {
    test_asset(
        "assets/totem_tower_troop_blue_idle_n.png",
        SpritesheetInfo {
            sprite_width: 296,
            sprite_height: 604,
            columns: 8,
            rows: 7,
            frame_count: 50,
        },
    );
}

#[test]
fn test_raid_mine_card_coins() {
    test_asset(
        "assets/raid_mine_card_coins.png",
        SpritesheetInfo {
            sprite_width: 383,
            sprite_height: 320,
            columns: 5,
            rows: 4,
            frame_count: 17,
        },
    );
}

#[test]
fn test_totem_tower_troop_red_idle_n() {
    test_asset(
        "assets/totem_tower_troop_red_idle_n.png",
        SpritesheetInfo {
            sprite_width: 296,
            sprite_height: 604,
            columns: 8,
            rows: 7,
            frame_count: 50,
        },
    );
}

#[test]
fn test_chest_rare_1_open_idle() {
    test_asset(
        "assets/chest_rare_1_open_idle.png",
        SpritesheetInfo {
            sprite_width: 942,
            sprite_height: 809,
            columns: 8,
            rows: 8,
            frame_count: 60,
        },
    );
}

#[test]
fn test_priest_troop_red_heal_unit_s() {
    test_asset(
        "assets/priest_troop_red_heal_unit_s.png",
        SpritesheetInfo {
            sprite_width: 320,
            sprite_height: 456,
            columns: 9,
            rows: 9,
            frame_count: 78,
        },
    );
}

#[test]
fn test_totem_tower_troop_red_eat_s() {
    test_asset(
        "assets/totem_tower_troop_red_eat_s.png",
        SpritesheetInfo {
            sprite_width: 296,
            sprite_height: 604,
            columns: 9,
            rows: 8,
            frame_count: 70,
        },
    );
}

#[test]
fn test_chest_common_4_call() {
    test_asset(
        "assets/chest_common_4_call.png",
        SpritesheetInfo {
            sprite_width: 942,
            sprite_height: 809,
            columns: 5,
            rows: 5,
            frame_count: 25,
        },
    );
}

#[test]
fn test_chest_rare_3_lvlup() {
    test_asset(
        "assets/chest_rare_3_lvlup.png",
        SpritesheetInfo {
            sprite_width: 942,
            sprite_height: 809,
            columns: 6,
            rows: 6,
            frame_count: 31,
        },
    );
}

#[test]
fn test_chest_rare_1_call() {
    test_asset(
        "assets/chest_rare_1_call.png",
        SpritesheetInfo {
            sprite_width: 942,
            sprite_height: 809,
            columns: 5,
            rows: 5,
            frame_count: 25,
        },
    );
}

#[test]
fn test_chest_common_4_open_idle() {
    test_asset(
        "assets/chest_common_4_open_idle.png",
        SpritesheetInfo {
            sprite_width: 942,
            sprite_height: 809,
            columns: 8,
            rows: 8,
            frame_count: 60,
        },
    );
}

#[test]
fn test_resized_chest_common_1_open_idle() {
    test_asset(
        "assets/resized_test/chest_common_1_open_idle.png",
        SpritesheetInfo {
            sprite_width: 512,
            sprite_height: 439,
            columns: 8,
            rows: 8,
            frame_count: 60,
        },
    );
}

#[test]
fn test_resized_monk_tower_troop_rock_shadow() {
    test_asset(
        "assets/resized_test/monk_tower_troop_rock_shadow.png",
        SpritesheetInfo {
            sprite_width: 197,
            sprite_height: 585,
            columns: 7,
            rows: 7,
            frame_count: 45,
        },
    );
}

#[test]
fn test_resized_totem_tower_troop_blue_eat_n() {
    test_asset(
        "assets/resized_test/totem_tower_troop_blue_eat_n.png",
        SpritesheetInfo {
            sprite_width: 250,
            sprite_height: 512,
            columns: 9,
            rows: 8,
            frame_count: 70,
        },
    );
}

#[test]
fn test_zap_card_projectile() {
    test_asset(
        "assets/zap_card_projectile.png",
        SpritesheetInfo {
            sprite_width: 1508,
            sprite_height: 106,
            columns: 2,
            rows: 12,
            frame_count: 23, // Expected 24 in index, but last frame is empty
        },
    );
}

#[test]
fn test_zap_card_hit() {
    test_asset(
        "assets/zap_card_hit.png",
        SpritesheetInfo {
            sprite_width: 103,
            sprite_height: 100,
            columns: 4,
            rows: 4,
            frame_count: 16, // Expected 13 in index, but last 3 frames are active
        },
    );
}

#[test]
fn test_resized_priest_troop_blue_heal_unit_n() {
    test_asset(
        "assets/resized_test/priest_troop_blue_heal_unit_n.png",
        SpritesheetInfo {
            sprite_width: 319,
            sprite_height: 455,
            columns: 9,
            rows: 9,
            frame_count: 78,
        },
    );
}
