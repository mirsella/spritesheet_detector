use spritesheet_detector::{SpritesheetInfo, analyze_spritesheet};

#[test]
fn test_asset_example() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/example.png").unwrap(), 40),
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
        analyze_spritesheet(&image::open("assets/map_tiles.png").unwrap(), 40),
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
        analyze_spritesheet(&image::open("assets/map_overlays.png").unwrap(), 40),
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
        analyze_spritesheet(&image::open("assets/map_tiles_borders.png").unwrap(), 40),
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
        analyze_spritesheet(&image::open("assets/bomb_card_area.png").unwrap(), 40),
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
        analyze_spritesheet(
            &image::open("assets/lightning_mage_card_overlay.png").unwrap(),
            40
        ),
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
            40
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
        analyze_spritesheet(
            &image::open("assets/ghoul_ripper_troop_hit.png").unwrap(),
            40
        ),
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
        analyze_spritesheet(
            &image::open("assets/totem_tower_totem_troop_vfx.png").unwrap(),
            40
        ),
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
            40
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
        analyze_spritesheet(
            &image::open("assets/totem_tower_totem_troop_active.png").unwrap(),
            40
        ),
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
        analyze_spritesheet(
            &image::open("assets/totem_tower_totem_troop_spawn.png").unwrap(),
            40
        ),
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
        analyze_spritesheet(
            &image::open("assets/totem_tower_totem_troop_death.png").unwrap(),
            40
        ),
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
        analyze_spritesheet(
            &image::open("assets/building_construction.png").unwrap(),
            40
        ),
        SpritesheetInfo {
            sprite_width: 248,
            sprite_height: 322,
            columns: 7,
            rows: 7,
            frame_count: 49,
        }
    );
}

#[test]
fn test_building_deconstruction() {
    assert_eq!(
        analyze_spritesheet(
            &image::open("assets/building_deconstruction.png").unwrap(),
            40
        ),
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
        analyze_spritesheet(&image::open("assets/building_destruction.png").unwrap(), 40),
        SpritesheetInfo {
            sprite_width: 340,
            sprite_height: 322,
            columns: 8,
            rows: 7,
            frame_count: 52,
        }
    );
}

#[test]
fn test_peasant_card_overlay() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/peasant_card_overlay.png").unwrap(), 40),
        SpritesheetInfo {
            sprite_width: 383,
            sprite_height: 392,
            columns: 11,
            rows: 10,
            frame_count: 110,
        }
    );
}

#[test]
fn test_necromancer_troop_projectile() {
    assert_eq!(
        analyze_spritesheet(
            &image::open("assets/necromancer_troop_projectile.png").unwrap(),
            40
        ),
        SpritesheetInfo {
            sprite_width: 77,
            sprite_height: 49,
            columns: 3,
            rows: 3,
            frame_count: 5,
        }
    );
}

#[test]
fn test_necromancer_troop_projectile_impact() {
    assert_eq!(
        analyze_spritesheet(
            &image::open("assets/necromancer_troop_projectile_impact.png").unwrap(),
            40
        ),
        SpritesheetInfo {
            sprite_width: 82,
            sprite_height: 74,
            columns: 3,
            rows: 3,
            frame_count: 5,
        }
    );
}

#[test]
fn test_bomber_troop_projectile() {
    assert_eq!(
        analyze_spritesheet(
            &image::open("assets/bomber_troop_projectile.png").unwrap(),
            40
        ),
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
        analyze_spritesheet(
            &image::open("assets/monk_tower_troop_rock_attack.png").unwrap(),
            40
        ),
        SpritesheetInfo {
            sprite_width: 238,
            sprite_height: 706,
            columns: 6,
            rows: 6,
            frame_count: 36,
        }
    );
}

#[test]
fn test_chest_notif() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/chest_notif.png").unwrap(), 40),
        SpritesheetInfo {
            sprite_width: 99,
            sprite_height: 117,
            columns: 3,
            rows: 3,
            frame_count: 7,
        }
    );
}
