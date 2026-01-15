#[cfg(test)]
mod tests {
    use spritesheet_detector::analyze_spritesheet;
    use std::path::PathBuf;

    fn get_asset_path(name: &str) -> PathBuf {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("assets");
        path.push("resized_test");
        path.push(name);
        path
    }

    #[test]
    fn inspect_resized_assets() {
        let assets = vec![
            "chest_common_1_open_idle.png",
            "monk_tower_troop_rock_shadow.png",
            "totem_tower_troop_blue_eat_n.png",
            "priest_troop_blue_heal_unit_n.png",
        ];

        for asset_name in assets {
            let path = get_asset_path(asset_name);
            println!("Inspecting: {:?}", path);
            let img = image::open(&path).expect("Failed to open image");
            let info = analyze_spritesheet(&img);
            println!("Result for {}: {:?}", asset_name, info);
        }
    }

    #[test]
    fn inspect_high_res_chest() {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("assets");
        path.push("chest_common_1_open_idle.png");
        println!("Inspecting High Res: {:?}", path);
        let img = image::open(&path).expect("Failed to open image");
        let info = analyze_spritesheet(&img);
        println!("Result for High Res Chest: {:?}", info);
    }
}
