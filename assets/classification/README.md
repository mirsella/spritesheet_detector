# Classification fixtures

These PNGs are unmodified copies of Mighty Minions source assets, not processor output. Their intended layouts come from the authored images and their use in the game. Paths below are relative to that game's `assets/` directory.

| Fixture | Source path |
| --- | --- |
| `lightning_mage_troop_projectile.png` | `cards/lightning_mage/lightning_mage_troop_projectile.png` |
| `source_zap_card_hit.png` | `cards/zap_card_hit.png` |
| `musketeer_troop_projectile_impact.png` | `cards/musketeer_lodge/musketeer_troop_projectile_impact.png` |
| `cursor_click_n.png` | `tutorial/cursor_click_n.png` |
| `settings_*.png` | `menu/settings_icons/`, same basenames |
| `feedback_*.png` | `game_ui/`, same basenames |
| `pause.png`, `restart.png` | `replay/`, same basenames |
| `envelope.png`, `voxride-wordmark.png` | `voxride/`, same basenames |
| `card_compare.png` | `menu/card_compare.png` |
| `card_*Frame.png` | `ui_card_frames/`, same basenames |
| `map_dark_forest_border_s_1.png` | `map/dark_forest/map_dark_forest_border_s_1.png` |
| `map_snow_territory_*_edge_s.png` | `map/snow/`, same basenames |

The four animation fixtures have these authored layouts:

| Fixture | Cell size | Grid | Active frames |
| --- | --- | --- | --- |
| `lightning_mage_troop_projectile.png` | 168 × 96 | 5 × 4 | 17 |
| `source_zap_card_hit.png` | 100 × 103 | 4 × 4 | 13 |
| `musketeer_troop_projectile_impact.png` | 90 × 89 | 2 × 2 | 4 |
| `cursor_click_n.png` | 500 × 580 | 2 × 1 | 2 |

The remaining fixtures are single images, including the complete map-border strip. The source Zap fixture differs from `assets/zap_card_hit.png`, which is 412 × 400 and has 16 active frames. Keep both tests.

The existing `UI_chest_lock.png` fixture has a 3 × 2 grid with six frames. Its dissimilar rows protect against treating every anticorrelated two-way split as static. Existing chest-idle and card-glow tests also protect repeated UI animation.
