# Data Access

The raw dataset (RGB images, per-class masks, COCO annotation JSON) is not committed to this
repository (`.gitignore` excludes `data/raw/`) -- it's large (the RGB image set alone is several
GB) and typically distributed separately.

Expected local layout: `data/raw/Set1_535/` -- see [RUNNING_THE_CODE.md](../RUNNING_THE_CODE.md#2-lay-out-the-raw-data)
for the exact structure.

If you receive the data as two separate archives (one with the RGB images, one with
`Json_Output/` + `Mask/`), extract both into the same destination folder -- they share the same
`Set1_535/` root directory and merge automatically.
