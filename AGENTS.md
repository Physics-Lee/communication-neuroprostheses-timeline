# AGENTS.md — `communication-neuroprostheses-timeline/`

This repository maintains the **Living Timeline of Communication Neuroprostheses for People with Paralysis (2021–present)**—a weekly-updated dataset (`studies.yaml`) and a self-contained two-pass Matplotlib layout/verification engine (`render_timeline.py`) that produces publication-grade raster (`.png`) and vector (`.pdf`) figures across three modality lanes (`speech`, `typing`, `handwriting`).

---

## Repository Contents

| File | Role | Edit Policy |
|---|---|---|
| [`studies.yaml`](studies.yaml) | **Single source of truth** for all study cards, participant IDs, dates, and cohort metadata (`participant_aliases`, `home_use_summary`). | **Primary file to edit** when adding/updating studies. |
| [`studies.json`](studies.json) | Auto-generated JSON mirror of `studies.yaml` (written automatically whenever `render_timeline.py` runs with PyYAML installed). | **Do NOT hand-edit.** Always regenerate via `python render_timeline.py`. |
| [`render_timeline.py`](render_timeline.py) | Self-contained two-pass layout solver, invariant gate checker, and PNG/PDF renderer. | Keep strictly **self-contained** (no external repo path dependencies). |
| [`weekly_patrol.py`](weekly_patrol.py) | Automated literature patrol (Europe PMC + arXiv) and preprint-to-journal graduation checker (Crossref `is-preprint-of`). | Edit only when refining search queries or report formatting. |
| [`communication_neuroprostheses_timeline.png`](communication_neuroprostheses_timeline.png) | Rendered 200 DPI raster figure (`27.0 × 19.2 in`). | **Generated artifact** — always re-render after changing `studies.yaml` or `render_timeline.py`. |
| [`communication_neuroprostheses_timeline.pdf`](communication_neuroprostheses_timeline.pdf) | Deterministic vector PDF (`SOURCE_DATE_EPOCH=946684800`, TrueType Type-42 font embedding `pdf.fonttype=42`). | **Generated artifact** — re-render alongside `.png`. |
| [`.github/workflows/weekly_update.yml`](.github/workflows/weekly_update.yml) | GitHub Actions workflow triggered on pushes to `studies.yaml` or `render_timeline.py` on `main`; re-renders `.png`/`.pdf` and commits `studies.json`. | Keep font package `fonts-dejavu-core` installed so CI bounding boxes match local runs. |
| [`README.md`](README.md) / [`README.zh-CN.md`](README.zh-CN.md) | English and Simplified Chinese documentation. | Keep both files synchronized when updating schema or rules. |

---

## Data Flow & How to Run

```text
studies.yaml  ──►  python render_timeline.py --yaml studies.yaml --out communication_neuroprostheses_timeline --check-overlaps
                        ├──► studies.json                                    (auto-synced JSON mirror)
                        ├──► communication_neuroprostheses_timeline.png      (200 DPI raster)
                        └──► communication_neuroprostheses_timeline.pdf      (Type-42 vector PDF)
```

### Standard Verification Command
Run from the repository root after **any** data or layout change:
```bash
python render_timeline.py --yaml studies.yaml --out communication_neuroprostheses_timeline --check-overlaps
```
Expected clean output:
```text
[ok] Rendered <N> cards -> .../communication_neuroprostheses_timeline.{png,pdf}
     Cohort: <M> unique participants (...)
     Overlap check: ALL CLEAN
```

---

## Load-Bearing Conventions & Invariants (Do Not Break)

### 1. Date Convention (`date` & `year_fraction`)
Every card in `studies.yaml` belongs to one of three entry types and must follow these exact date rules:
1. **Published Journal Articles**: Use the journal's official **Issue Date** (for online-only journals without print issues, such as *Nat. Commun.* or *Adv. Sci.*, use the sole **Article Publication Date**). Citation format: `"Author et al., YYYY"` (with `a`/`b` suffix if disambiguating same-author/same-year studies, or initial prefix like `"Y. Zhang et al., 2026"`).
2. **Preprints & Conference Papers**: Use the preprint posting date (`bioRxiv` / `medRxiv` / `arXiv`) or the conference proceedings publication date (`Interspeech`, `NeurIPS`, etc.).
3. **Company Announcements (Non-Paper Releases)**: Use the official announcement release date, and format `citation` directly as `"Company, yyyy.mm.dd"` (e.g., `"Neuralink, 2026.01.28"`, `"Paradromics, 2026.09.14"`). Store the link in `url:` instead of `doi:`.

- **`year_fraction` overrides `date` for x-placement**: `render_timeline.py` reads `float(entry.get("year_fraction") or date_to_year_fraction(entry["date"]))`. If you update an entry's `date`, either update `year_fraction` to match `round(year + day_of_year / days_in_year, 2)` or omit `year_fraction` so it is computed automatically.

### 2. Six Orthogonal Visual Dimensions (`studies.yaml` Schema)
`load_studies()` validates every entry against strict enums and fails fast on unknown values:
- `lane`: `speech` | `typing` | `handwriting`
- `data_availability`: `open` (`#00A087` teal, placed **directly on** the track line) | `on_request` (`#c07f00` amber) | `closed` (`#cc3324` red) | `preprint` (`#7F7F7F` gray) | `reanalysis` (`#7B3294` purple)
- `decoding_mode`: `online` (solid marker = closed-loop participant demo) | `offline` (hollow white center with colored ring = offline analysis / benchmark reanalysis)
- `electrode`: `ecog` (`o` circle) | `utah` (`s` square) | `thin_film` (`h` hexagon) | `microwire` (`v` down-triangle)
- `participants`: non-empty list of strings; **every string must end with `▲` (`\u25b2`, natural speech intact) or `△` (`\u25b3`, vocal paralysis / dysarthria / anarthria / LIS)**.
  - Re-implant aliases (e.g., `B1r: B1` for Bravo-1's re-implant) live under `metadata.participant_aliases` in `studies.yaml` so unique cohort totals (`total_unique` and per-lane `n = ... unique participants`) are deduplicated automatically.
  - Optional `n_override: <int>` overrides the displayed `n=` count when a study's participant count differs from `len(participants)`.
- `task_instruction`: `inner` (`◆`) | `intended` (`◇`) | `both` (`◆/◇`) — appended automatically to Line 1 (`citation`).
- `decoded_output`: `voice` (`⬟`) | `text` (`⬠`) | `both` (`⬟/⬠`) — appended automatically after `descriptor` on Line 2.

### 3. Four-Item `metrics` Block & Card Line 2 Formatting
Every entry must define all four keys under `metrics:`:
- `vocab_size`: e.g., `"vocab 125k"`, `"open vocab"`, `"vocab 50"`, `"vocab n/r"`
- `sentences`: corpus / trial volume description (stored in YAML/JSON for dataset completeness; not printed on the compact figure card)
- `wer`: headline error rate string (e.g., `"WER 23.9%"`, `"listener WER 20%"`, `"CER 8.9% (WER n/r)"`)
- `speed`: communication rate (e.g., `"62 wpm"`, `"110 cpm"`, `"n/r"`)

**Card Line 2 switches**:
- By default, `speed` is printed on the card whenever `speed != "n/r"`. Set `card_show_speed: false` on an entry if `speed` is already mentioned inside `descriptor` (e.g., `descriptor: "78 wpm + avatar"`) or if omitting speed on the card is needed to keep the card compact.
- Optional `sep_after_output: "; "` (default `", "`) customizes the punctuation immediately following the `⬟`/`⬠` glyph.

### 4. Hard Verification Gates in `render_timeline.py`
`render_timeline()` enforces five hard assertions before writing output files:
1. **Gate 1 (Output glyph & listener consistency)**: Every card's Line 2 contains `⬟` or `⬠` (and Line 1 contains neither). Furthermore, **any card whose metric line mentions `"listener"` must have `decoded_output: voice` or `both` (`⬟`)**.
2. **Gate 2 (Task-instruction glyph)**: Every card's Line 1 ends with `◆` or `◇`.
3. **Gate 3 (Green-on-track & marker separation)**: Only `data_availability: open` markers sit on the track line (`y == TRACKS[lane]`); all non-open markers hang `0.52 * level` units toward their label side (`above` or `below`), stepping outward in `0.52` increments whenever two markers on the same level are within `< 0.045` years on the x-axis.
4. **Gate 4 (Equalized 6-legend row)**: The 6 top-left legend boxes are redistributed in pixel space so all 5 inter-legend horizontal gaps agree to within `<= 0.5 px`, and no card on the top `speech` row may intrude left of `LEG_X_END = x0 + 0.56 * (x1 - x0)`.
5. **Gate 5 (Zero bounding-box overlap)**: `check_figure_overlaps(fig)` checks every pairwise text and legend patch bounding box in rendered pixel coordinates and raises `RuntimeError` if any pair overlaps by `> 1 px` in both x and y.

### 5. Layout Solver Quirks to Keep in Mind
- **Above/Below side assignment (`SIDE[i]`) vs. packing order (`items` order)**:
  - `SIDE[i]` (whether a card prefers `below` or `above` the track line) is assigned by sorting each lane's studies chronologically by `xm` (`year_fraction`).
  - However, `layout()` packs cards in **their YAML list order** (`for i, ... in enumerate(items)`), first trying `abs(cx - xm) <= 0.45` and then relaxing to `MAXSHIFT = 2.05`. In `typing` and `handwriting`, if the preferred side is full, `layout()` can fall back to the opposite side. That is why `neuralink2026_p7` is intentionally placed after `jude2026a` at the bottom of `studies.yaml` (see comment in `studies.yaml:698-699`).
- **When a new card fails to pack (`AssertionError: Could not pack card ...`)**:
  1. Check whether `descriptor` or `wer` text can be tightened or `card_show_speed: false` should be set.
  2. If the timeline extends into a new year and the right margin is tight, `x1 = max(2029.55, max_xm + 2.74)` in `render_timeline.py` scales automatically with `max_xm`, or an additional y-row can be added to `ROWS[lane]`.
