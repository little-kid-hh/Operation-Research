This seed candidate intentionally uses a small set of explainable
item-distribution features:

- `dominant_type_share` captures repetition versus heterogeneity at
  dispatch level.
- `p90_long_over_bin_long` and `p90_mid_over_bin_mid` emphasize upper-tail
  dimension pressure instead of means.
- `thin_item_share` captures whether many items have one very small side,
  which can help packing flexibility.
- `max_face_area_load_over_floor` approximates aggregate face-pressure
  against the vehicle floor footprint.
- `tight_bin_large_piece_interaction` is an explicit interaction term meant
  to linearize a regime that trees can usually capture more naturally.

