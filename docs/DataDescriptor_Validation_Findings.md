# AerialYield-T²M Draft: Validation Findings & Filled-In Values

This document cross-references the tomato draft (`AerialYield-T2M_Tomato_DataDescriptor.pdf`)
against its sibling, published blueberry paper (`AerialYield_B2.pdf`, `AerialYield-B2D`), and fills
in the draft's `[TO BE CALCULATED]` / `[TO BE CONFIRMED]` placeholders with real numbers computed
directly from the `Set1_535` raw data on disk (`data/raw/Set1_535/`, 535 images) using this
project's pipeline (`scripts/prepare_annotations.py`, `scripts/technical_validation.py`).

**Read the three critical findings below before using any of the filled-in numbers** — they
contradict core assumptions in the draft's Methods section and must be resolved with the research
team before submission.

> **Update:** an independent audit (`audit_results/audit_report.md`, `claim_verification.csv`,
> `remediation_plan.md`) has since re-derived these same numbers from raw files and found
> additional issues this document doesn't cover: 908 self-intersecting COCO polygons, duplicate
> annotation/crop pairs, and a precise correction to the mask-component figures below (this
> document's "19,157" is one of *two* different numbers — the current code actually produces
> 19,159; 19,157 is a separate, corrected recomputation — see the audit for the exact distinction).
> It independently confirms the split-leakage finding in §4. One claim in the audit's own prose —
> that all images carry "EXIF orientation tag 1" — was checked again for this update and does not
> hold: `Image.getexif()`, `Image._getexif()`, and the raw `im.info` all agree these files carry no
> EXIF segment at all (bare JFIF headers), so this document's original "no EXIF" finding stands.
> Treat `audit_results/` as the more rigorous source for anything about geometry/annotation
> validity and provenance; treat this document as the more rigorous source for the acquisition-
> modality and taxonomy discrepancies (findings 1–3 below), which the audit did not dispute.

---

## Critical findings (resolve these first)

### 1. This data is not "535 mobile-phone images" — it is video-frame-extracted imagery

The draft's Methods section states the 535-image subset was acquired with mobile-phone cameras at
close range, with EXIF-based device metadata. The actual files contradict this:

- **Zero EXIF metadata** in all 535 images (no camera model, no orientation tag — confirmed on
  every single file, not just a sample).
- Filenames follow `YYYYMMDD_HHMMSS_FFFFFF.jpg`, and the frame-number suffix (`FFFFFF`) increases
  by a near-constant **10** across almost every consecutive file (occasional gaps of 11, 12, 20,
  or 30 — consistent with a dropped/skipped frame during extraction, not with independent photos).
- All 535 images fall into exactly **3 filename timestamps, all on the same calendar day**
  (2026-04-16, at 12:31:09, 12:33:30 and 12:34:59) — i.e. three short continuous clips captured
  minutes apart, not a walkthrough of independent photographs over a shooting session.
- Image resolution is uniformly **7680×4320 (landscape) or 4320×7680 (portrait)** — exactly the
  resolution the *blueberry* paper (`AerialYield-B2D`, Table 9) assigns to its **`video_frame`**
  modality (67 images, "added 2026-02-12 video-frame subset"), not its smartphone modality
  (3000×4000).

**Conclusion: `Set1_535` reads as video-frame-extracted imagery (likely from one or a few short
video clips), structurally the same *kind* of data as the blueberry paper's small `video_frame`
subset — not 535 independent mobile-phone stills.** This affects the Abstract, the entire
"Mobile-phone acquisition procedure" subsection, Table 6, Table 9, and the EXIF-orientation part
of Technical Validation. It should be corrected with the research team before anything is
submitted, ideally by asking whoever collected the data what device/software actually produced
these 535 files.

### 2. Only 535 images exist on disk — the drone subset (150 images) has not been delivered

The draft's abstract and every table assume **685** images (535 mobile + 150 drone). Only the
535-image `Set1_535` archive has been provided to this project; there is no drone/aerial subset
in `data/raw/`. Every number below is computed on **535 images only**. If the 150-image drone
subset exists, it needs to be added to `data/raw/` and this validation re-run before the
685-image totals in the draft can be filled in for real.

### 3. The draft assumes 4 ripeness classes; the actual annotation data has 6

The draft's Table 1 proposes a **4-class** merged scheme (`green`, `breaker`, `turning_pink`,
`red_ripe`), explicitly merging USDA turning+pink and light-red+red. The real annotation file
(`Json_Output/Set1_535.json`, COCO format) and the real `Mask/` folder define **6 classes**
matching the Silal *Quality Standard Guide For Tomato* exactly: `Green`, `Breakers`, `Turning`,
`Pink`, `Light Red`, `Red`. This is a genuine scientific choice for the authors, not something
data can resolve: either (a) adopt the 6-class scheme to match the source annotations and the
Silal guide directly, dropping the merge, or (b) keep the 4-class merge and programmatically
collapse `Turning`+`Pink` → `turning_pink` and `Light Red`+`Red` → `red_ripe` when building the
release. All numbers below are reported in the **real 6-class scheme**; §4 gives the 4-class
collapse for comparison if the team wants to keep the draft's original taxonomy.

### 4. The current train/val/test split leaks near-duplicate frames across subsets

This is a finding from *our own* pipeline (`configs/tomato_benchmark.yaml`), not the draft, but it
matters for the same reason the draft's own "Split-generation strategy" section warns about:
consecutive video frames are not independent. Checking the realised split: **46.6% of
adjacent same-session frame pairs (248 of 532) land in different train/val/test subsets.** Every
one of the three sessions has train, val, *and* test images interleaved within a single
continuous clip:

| Session | train | val | test |
|---|---|---|---|
| 2026-04-16 12:31:09 | 147 | 31 | 33 |
| 2026-04-16 12:33:30 | 113 | 18 | 21 |
| 2026-04-16 12:34:59 | 114 | 31 | 27 |

If frames 10 apart in the source video are visually similar (likely at typical frame rates), this
is real train/test leakage that would inflate any reported benchmark metric. **The current
`tomato_benchmark.yaml` split should be replaced with a session-grouped split** (group by the
`YYYYMMDD_HHMMSS` session prefix, never split a session across subsets) before any benchmark
numbers are reported — this is exactly the grouping strategy the T²M draft itself specifies should
be used, and it has not yet been applied to `Set1_535`. Let me know if you'd like this implemented
now; it's a small change to `annotations.py::_safe_split`.

---

## Filled-in tables (535 images, real 6-class taxonomy, computed from `Set1_535`)

### Table 6 equivalent — acquisition source and metadata summary

| Source modality | Images | Capture device | Image geometry | Metadata source |
|---|---|---|---|---|
| Unconfirmed — file evidence indicates video-frame extraction, not "mobile" (see Finding 1) | 535 | Unknown — no EXIF present | 363 images 7680×4320 (landscape), 172 images 4320×7680 (portrait) | None embedded; date/time is filename-derived only (`YYYYMMDD_HHMMSS_frame.jpg`) |

### Table 9 equivalent — acquisition sessions (filename-derived; this is the *real* substitute for "EXIF orientation distribution", since no EXIF exists)

| Date | Time | Frames | Frame # range | Modal frame interval | Notes |
|---|---|---|---|---|---|
| 2026-04-16 | 12:31:09 | 211 | 9–2109 | 10 | 210/210 gaps are exactly 10 |
| 2026-04-16 | 12:33:30 | 152 | 3–1563 | 10 | 147×10, 3×20, 1×30 |
| 2026-04-16 | 12:34:59 | 172 | 9–1778 | 10 | 140×10, 27×11, 1×12, 3×20 |

EXIF orientation tag distribution: **535/535 "none" — no EXIF orientation tag present in any
file.**

### Table 7 equivalent — class instance counts and image coverage (real COCO annotations, the
authoritative instance source — see the dataset audit for why mask connected-components are not
used for this)

| Class | Instances | Share | Images containing the class |
|---|---:|---:|---:|
| Green | 6,558 | 70.82% | 534 / 535 |
| Breakers | 127 | 1.37% | 100 / 535 |
| Turning | 623 | 6.73% | 286 / 535 |
| Pink | 422 | 4.56% | 272 / 535 |
| Light Red | 1,409 | 15.22% | 486 / 535 |
| Red | 121 | 1.31% | 95 / 535 |
| **Total** | **9,260** | 100% | 535 / 535 |

### Table 8 equivalent — mask-pixel area and connected-component distribution

| Class | Mask pixels | Mask area % of all image area | Connected components* |
|---|---:|---:|---:|
| Green | 1,101,163,030 | 6.20% | 12,563 |
| Breakers | 37,003,205 | 0.21% | 317 |
| Turning | 140,034,431 | 0.79% | 1,403 |
| Pink | 85,386,279 | 0.48% | 1,021 |
| Light Red | 297,341,856 | 1.68% | 3,573 |
| Red | 21,891,236 | 0.12% | 282 |
| **Total foreground** | **1,682,820,037** | **9.48%** | — |

\* **Caution:** connected-component counts are inflated by lossy-JPEG mask compression artifacts
that split single tomatoes into multiple mask blobs (documented in
`docs/Preprocessing_Pipeline.md` and `outputs/tomato_benchmark_535/00_data_audit/dataset_audit.md`).
Report the **COCO annotation counts (Table 7) as the real instance counts**; report these
component counts only as a mask-quality QA figure, exactly as this project's own pipeline now does.

### Recommended split (as currently generated — see Finding 4 above before trusting this for benchmarking)

| Split | Images | Total instances | Green | Breakers | Turning | Pink | Light Red | Red |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 374 | 6,302 | 4,401 | 86 | 444 | 290 | 991 | 90 |
| val | 80 | 1,504 | 1,088 | 17 | 102 | 61 | 214 | 22 |
| test | 81 | 1,454 | 1,069 | 24 | 77 | 71 | 204 | 9 |
| **Total** | **535** | **9,260** | 6,558 | 127 | 623 | 422 | 1,409 | 121 |

### Image-quality screening summary (Table 10 equivalent)

Computed on downsampled (512px-wide), orientation-neutral copies, n = 535:

| Metric | Mean | Std (across images) | Min | Max |
|---|---:|---:|---:|---:|
| Brightness mean | 111.17 | 9.11 | 84.00 | 140.48 |
| Brightness std (within-image) | 50.75 | 4.75 | 37.83 | 63.26 |
| Saturation mean | 109.92 | 17.78 | 49.12 | 165.76 |
| Laplacian variance (sharpness) | 750.87 | 292.41 | 40.57 | 1789.20 |

- Images flagged low-sharpness (bottom 5% of Laplacian variance): **27**
- Images flagged extreme brightness (outer 2% tails): **22**

(These flag thresholds are illustrative/percentile-based, not the exact method the blueberry paper
used — the team should decide the final flagging rule before publication.)

### File integrity

- SHA-256: **535/535 unique — no exact duplicates.**
- Near-duplicate (perceptual-hash) screening: **not yet run** — given the video-frame-extraction
  finding above (Finding 1), this is now more important than originally scoped, since frames 10
  apart in a short clip may well be near-duplicates. Flagging as a follow-up rather than
  guessing; it's a ~15-20 minute compute job on this hardware.
- Mask completeness: every one of the 535 images has all 6 class masks + 1 overall mask (no
  missing artifacts — this matches the blueberry paper's "no missing image-mask pairs" finding).
- Class-overlap pixels (two class masks claiming the same pixel): present in **5 / 535** images,
  13,930 pixels total — negligible relative to ~9.5 billion total analyzed pixels.
- Class-union vs. overall-mask mismatch: **297 / 535** images show some discrepancy (median 13 px,
  max 9,592 px against a ~33.2-million-pixel image) — small in magnitude but present in over half
  the images; worth a one-line mention in Technical Validation rather than claiming a perfect
  match.

### Release size

- Raw data (`data/raw/Set1_535/`): **4.6 GB, 4,287 files** (535 images + masks + COCO JSON).
- Derived products from this pipeline (`outputs/tomato_benchmark_535/`): **1.2 GB, 10,913 files**
  (crops, semantic masks, splits, COCO/YOLO exports, audit figures).

---

## §4 — 4-class collapse, if the team wants to keep the draft's original taxonomy

If the 4-class scheme in the current draft is intentional (not just a placeholder), here is the
same Table 7 collapsed as the draft's Table 1 specifies (`turning_pink` = Turning+Pink,
`red_ripe` = Light Red+Red):

| Class (draft scheme) | Instances | Share |
|---|---:|---:|
| green | 6,558 | 70.82% |
| breaker | 127 | 1.37% |
| turning_pink | 1,045 (623+422) | 11.29% |
| red_ripe | 1,530 (1,409+121) | 16.52% |

---

## Appendix A checklist — resolution status

Cross-referencing the draft's own Table 12 ("Checklist of information required from the research
team before submission"):

| Item | Status |
|---|---|
| Total annotated instances and per-class counts | **Resolved** (real data, 535-image subset only — see Table 7 above) |
| Mask-pixel areas and connected-component counts | **Resolved** (Table 8 above; caveat on component-count reliability noted) |
| Image-quality metric summaries | **Resolved** (Table 10 above) |
| Preprocessing audit outputs (overlap, mismatch, duplicates, orientation) | **Resolved** (this document, "File integrity" section) |
| Realised split file lists and leakage-check results | **Partially resolved** — a split exists but is **not leakage-safe** (Finding 4); needs to be regenerated group-wise before it can be called "realised" in the paper's sense |
| Release file count and archive size | **Resolved** (this document) |
| Final class list and `class_map.json` | **Needs a decision, not data** — 6-class real data vs. 4-class draft scheme (Finding 3) |
| Confirmation annotation was fully manual | **Still outstanding** — not verifiable from file contents alone |
| Annotation provider, annotator count, review rate | **Still outstanding** — not in the delivered files |
| Site name, location, cultivar, planting date, etc. | **Still outstanding** — not in the delivered files (note: the sibling blueberry dataset's site was "Silal Al Foah Farm, Al Ain" — plausible but **not confirmed** for this tomato data; do not assume) |
| Acquisition date range | **Partially resolved** — filename evidence gives a single date, 2026-04-16, for all 535 images; whether more sessions exist in the missing 150-image drone subset is unknown |
| Mobile device model(s) / drone model / flight parameters | **Still outstanding**, and Finding 1 suggests the "mobile device" framing itself needs revisiting |
| Video frame-extraction interval | **Resolved from file evidence**: modal interval of 10 frame-numbers between consecutive files (Table above) — but this is inferred, not confirmed by whoever ran the extraction |
| Baseline experiment results (Table 11) | **Not run** — this project has only smoke-tested the training mechanism (1 epoch, 12 samples), which is not a reportable baseline; a real run needs the leakage-safe split from Finding 4 first |
| Author list, funding, licence, repository DOI, competing interests | **Outstanding — human input only, no data can resolve these** |

---

*Generated from a live analysis of `data/raw/Set1_535/` and this repository's pipeline outputs.
Re-run `scripts/technical_validation.py` after any changes to the raw data (e.g. if the 150-image
drone subset is added) to refresh these numbers.*
