# G0 data sanity report

- people: 903253
- with_origin: 519534
- citizenship_eligible: 419510
- citizenship_leaky_dropped: 4
- sport_eligible: 401563
- sport_leaky_dropped: 0
- field_eligible: 16487
- field_leaky_dropped: 0
- citizenship_cells_selected: 30
- citizenship_origins_selected: 13
- sport_cells_selected: 30
- sport_origins_selected: 13
- field_cells_selected: 30
- field_origins_selected: 11
- items: 900

Check: every slot has >= 10 selected cells, >= 5 origins and close to items_per_slot items.
If not, fix QIDs or thresholds in configs/g0.yaml and log it in notes/decisions.md.