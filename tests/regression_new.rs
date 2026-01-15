use spritesheet_detector::{SpritesheetInfo, analyze_spritesheet};

#[test]
fn test_regression_new_totem_tower_troop_blue_eat_n() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/totem_tower_troop_blue_eat_n.png").unwrap()),
        SpritesheetInfo {
            sprite_width: 296,
            sprite_height: 604,
            columns: 9,
            rows: 8,
            frame_count: 70,
        }
    );
}

#[test]
fn test_regression_new_priest_troop_blue_heal_unit_n() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/priest_troop_blue_heal_unit_n.png").unwrap()),
        SpritesheetInfo {
            sprite_width: 320,
            sprite_height: 456,
            columns: 9,
            rows: 9,
            frame_count: 78,
        }
    );
}

#[test]
fn test_regression_new_chest_common_0_lvlup() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/chest_common_0_lvlup.png").unwrap()),
        SpritesheetInfo {
            sprite_width: 942,
            sprite_height: 809,
            columns: 6,
            rows: 6,
            frame_count: 31,
        }
    );
}

#[test]
fn test_regression_new_chest_common_1_call() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/chest_common_1_call.png").unwrap()),
        SpritesheetInfo {
            sprite_width: 942,
            sprite_height: 809,
            columns: 5,
            rows: 5,
            frame_count: 25,
        }
    );
}

#[test]
fn test_regression_new_monk_tower_troop_rock_attack() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/monk_tower_troop_rock_attack.png").unwrap()),
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
fn test_regression_new_peasant_card_overlay() {
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
fn test_regression_new_totem_tower_troop_blue_idle_n() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/totem_tower_troop_blue_idle_n.png").unwrap()),
        SpritesheetInfo {
            sprite_width: 296,
            sprite_height: 604,
            columns: 8,
            rows: 7,
            frame_count: 50,
        }
    );
}

#[test]
fn test_regression_new_chest_common_1_open_idle() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/chest_common_1_open_idle.png").unwrap()),
        SpritesheetInfo {
            sprite_width: 942,
            sprite_height: 809,
            columns: 8,
            rows: 8,
            frame_count: 60,
        }
    );
}

#[test]
fn test_regression_new_punk_band_guitarrist_troop_blue_death() {
    assert_eq!(
        analyze_spritesheet(
            &image::open("assets/punk_band_guitarrist_troop_blue_death.png").unwrap()
        ),
        SpritesheetInfo {
            sprite_width: 340,
            sprite_height: 380,
            columns: 4,
            rows: 4,
            frame_count: 16,
        }
    );
}

// New tests added in this iteration

#[test]
fn test_regression_new_punk_band_guitarrist_troop_red_death() {
    assert_eq!(
        analyze_spritesheet(
            &image::open("assets/punk_band_guitarrist_troop_red_death.png").unwrap()
        ),
        SpritesheetInfo {
            sprite_width: 340,
            sprite_height: 380,
            columns: 4,
            rows: 4,
            frame_count: 16,
        }
    );
}

#[test]
fn test_regression_new_raid_mine_card_coins() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/raid_mine_card_coins.png").unwrap()),
        SpritesheetInfo {
            sprite_width: 383,
            sprite_height: 320,
            columns: 5,
            rows: 4,
            frame_count: 17,
        }
    );
}

#[test]
fn test_regression_new_totem_tower_troop_red_idle_n() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/totem_tower_troop_red_idle_n.png").unwrap()),
        SpritesheetInfo {
            sprite_width: 296,
            sprite_height: 604,
            columns: 8,
            rows: 7,
            frame_count: 50,
        }
    );
}

#[test]
fn test_regression_new_chest_rare_1_open_idle() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/chest_rare_1_open_idle.png").unwrap()),
        SpritesheetInfo {
            sprite_width: 942,
            sprite_height: 809,
            columns: 8,
            rows: 8,
            frame_count: 60,
        }
    );
}

#[test]
fn test_regression_new_priest_troop_red_heal_unit_s() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/priest_troop_red_heal_unit_s.png").unwrap()),
        SpritesheetInfo {
            sprite_width: 320,
            sprite_height: 456,
            columns: 9,
            rows: 9,
            frame_count: 78,
        }
    );
}

#[test]
fn test_regression_new_totem_tower_troop_red_eat_s() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/totem_tower_troop_red_eat_s.png").unwrap()),
        SpritesheetInfo {
            sprite_width: 296,
            sprite_height: 604,
            columns: 9,
            rows: 8,
            frame_count: 70,
        }
    );
}

#[test]
fn test_regression_new_chest_common_4_call() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/chest_common_4_call.png").unwrap()),
        SpritesheetInfo {
            sprite_width: 942,
            sprite_height: 809,
            columns: 5,
            rows: 5,
            frame_count: 25,
        }
    );
}

#[test]
fn test_regression_new_chest_rare_3_lvlup() {
    assert_eq!(
        analyze_spritesheet(&image::open("assets/chest_rare_3_lvlup.png").unwrap()),
        SpritesheetInfo {
            sprite_width: 942,
            sprite_height: 809,
            columns: 6,
            rows: 6,
            frame_count: 31,
        }
    );
}
