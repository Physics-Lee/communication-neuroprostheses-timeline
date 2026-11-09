#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone renderer for the Living Timeline of Communication Neuroprostheses.

Reads `studies.yaml` (and auto-syncs `studies.json`), validates all 6 visual-
encoding dimensions and the 4-item `metrics` schema (`vocab_size`, `sentences`,
`wer`, `speed`), lays out cards via a two-pass measured-text solver with
automatic collision avoidance, verifies zero text/legend bounding-box overlaps,
and outputs both raster (`.png`, 200 DPI) and vector (`.pdf`, TrueType Type-42
embedded) figures.

Usage:
    python render_timeline.py
    python render_timeline.py --yaml studies.yaml --out communication_neuroprostheses_timeline --check-overlaps
"""
import argparse
import datetime as dt
import itertools
import json
import math
import os
import sys
from typing import Any, Optional

os.environ.setdefault("SOURCE_DATE_EPOCH", "946684800")  # deterministic PDF

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
try:
    import yaml
except ImportError:
    yaml = None

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "pdf.fonttype": 42,
})

LANE_SPEECH, LANE_TYPE, LANE_HAND = "speech", "typing", "handwriting"

OPEN_C = {
    2: "#00A087",    # open data (NPG teal)
    1: "#c07f00",    # data on request (amber)
    0: "#cc3324",    # closed data (red)
    "P": "#7F7F7F",  # preprint (neutral gray)
    "R": "#7B3294",  # reanalysis on open data (purple)
}

ELECTRODE_MAP = {
    "ecog": "o",
    "utah": "s",
    "thin_film": "h",
    "microwire": "v",
}

AVAILABILITY_MAP = {
    "open": "open",
    "on_request": "partial",
    "partial": "partial",
    "closed": "closed",
    "preprint": "unknown",
    "unknown": "unknown",
    "reanalysis": "reanalysis",
}

TASK_GLYPH = {
    "inner": "\u25c6",          # ◆
    "intended": "\u25c7",       # ◇
    "both": "\u25c6/\u25c7",    # ◆/◇
}

OUTPUT_GLYPH = {
    "voice": "\u2b1f",          # ⬟
    "text": "\u2b20",           # ⬠
    "both": "\u2b1f/\u2b20",    # ⬟/⬠
}


def date_to_year_fraction(date_str: str) -> float:
    """Convert an ISO calendar date string into a decimal year fraction.

    Parameters
    ----------
    date_str : str
        Calendar date formatted as ``YYYY-MM-DD``.

    Returns
    -------
    float
        Decimal year rounded to two decimal places (e.g., ``2021.36``).
    """
    d = dt.date.fromisoformat(str(date_str))
    doy = d.timetuple().tm_yday
    days_in_year = 366 if (d.year % 4 == 0 and (d.year % 100 != 0 or d.year % 400 == 0)) else 365
    return round(d.year + doy / days_in_year, 2)


def normalize_participant_id(token: str, aliases: dict[str, str]) -> str:
    """Strip speech-ability suffixes and resolve re-implant participant aliases.

    Parameters
    ----------
    token : str
        Raw participant token ending with ``▲`` (natural speech intact) or
        ``△`` (vocal paralysis), such as ``"T15△"`` or ``"B1r△"``.
    aliases : dict of str to str
        Mapping from re-implant participant IDs to canonical individual IDs
        (e.g., ``{"B1r": "B1"}``).

    Returns
    -------
    str
        Canonical participant identifier used for unique cohort deduplication.
    """
    raw = token.replace("\u25b2", "").replace("\u25b3", "").strip()
    return aliases.get(raw, raw)


def load_studies(
    yaml_path: str, sync_json: bool = True
) -> tuple[list[tuple[str, float, str, str, str, str, str]], dict[str, Any]]:
    """Parse ``studies.yaml`` (or fallback ``studies.json``) and compute cohort totals.

    Parameters
    ----------
    yaml_path : str
        Path to the primary ``studies.yaml`` dataset (or ``studies.json`` fallback).
    sync_json : bool, default=True
        If ``True`` and PyYAML is available, automatically write a synchronized
        ``studies.json`` mirror alongside ``yaml_path``.

    Returns
    -------
    items : list of tuple
        Seven-tuples ``(name_line, xm, shape, opn, metric_line, lane, mode)``
        consumed by the two-pass layout solver.
    counts_summary : dict of str to Any
        Cohort summary and structured study records containing ``lane_totals``,
        ``total_unique``, ``home_summary``, ``study_records``, and ``metadata``.

    Raises
    ------
    ValueError
        If any entry violates the six orthogonal visual-encoding enums, participant
        triangle suffix convention, or required four-item ``metrics`` schema.
    """
    if yaml is not None and os.path.exists(yaml_path):
        with open(yaml_path, "r", encoding="utf-8") as f:
            doc = yaml.safe_load(f)
        if sync_json and yaml_path.endswith((".yaml", ".yml")):
            json_path = os.path.splitext(yaml_path)[0] + ".json"
            with open(json_path, "w", encoding="utf-8", newline="\n") as jf:
                json.dump(doc, jf, ensure_ascii=False, indent=2)
                jf.write("\n")
    else:
        json_path = (os.path.splitext(yaml_path)[0] + ".json"
                     if yaml_path.endswith((".yaml", ".yml")) else yaml_path)
        with open(json_path, "r", encoding="utf-8") as f:
            doc = json.load(f)

    meta = doc.get("metadata", {})
    aliases = meta.get("participant_aliases", {"B1r": "B1"})
    home_summary = meta.get("home_use_summary", "1 used a system at home (T15)")

    items = []
    study_records = []
    lane_sets = {LANE_SPEECH: set(), LANE_TYPE: set(), LANE_HAND: set()}
    all_participants = set()

    required_metric_keys = ("vocab_size", "sentences", "wer", "speed")

    for idx, entry in enumerate(doc["studies"]):
        cit = entry["citation"].strip()
        task_k = entry["task_instruction"]
        if task_k not in TASK_GLYPH:
            raise ValueError(f"{entry['id']}: invalid task_instruction {task_k!r}")
        name_line = f"{cit} {TASK_GLYPH[task_k]}"

        xm = float(entry.get("year_fraction") or date_to_year_fraction(entry["date"]))
        elec_k = entry["electrode"]
        if elec_k not in ELECTRODE_MAP:
            raise ValueError(f"{entry['id']}: invalid electrode {elec_k!r}")
        shape = ELECTRODE_MAP[elec_k]

        avail_k = entry["data_availability"]
        if avail_k not in AVAILABILITY_MAP:
            raise ValueError(f"{entry['id']}: invalid data_availability {avail_k!r}")
        opn = AVAILABILITY_MAP[avail_k]

        out_k = entry["decoded_output"]
        if out_k not in OUTPUT_GLYPH:
            raise ValueError(f"{entry['id']}: invalid decoded_output {out_k!r}")
        out_glyph = OUTPUT_GLYPH[out_k]

        parts = entry["participants"]
        for p in parts:
            if not (p.endswith("\u25b2") or p.endswith("\u25b3")):
                raise ValueError(
                    f"{entry['id']}: participant {p!r} must end with ▲ (natural speech) or △ (vocal paralysis)"
                )
        n_count = entry.get("n_override", len(parts))
        part_str = "/".join(parts)
        sep = entry.get("sep_after_output", ", ")
        desc = entry["descriptor"].strip()

        m_obj = entry["metrics"]
        if isinstance(m_obj, dict):
            missing = [k for k in required_metric_keys if k not in m_obj]
            if missing:
                raise ValueError(f"{entry['id']}: metrics missing keys {missing}")
            vocab_s = str(m_obj["vocab_size"]).strip()
            wer_s = str(m_obj["wer"]).strip()
            speed_s = str(m_obj["speed"]).strip()
            show_speed = entry.get("card_show_speed", speed_s.lower() != "n/r")
            fields = ([speed_s] if show_speed else []) + [wer_s, vocab_s]
            metrics_str = ", ".join(fields)
        else:
            metrics_str = str(m_obj).strip()

        metric_line = f"{desc} {out_glyph}{sep}{metrics_str} (n={n_count}, {part_str})"

        lane = entry["lane"]
        if lane not in lane_sets:
            raise ValueError(f"{entry['id']}: invalid lane {lane!r}")
        mode = entry["decoding_mode"]
        if mode not in ("online", "offline"):
            raise ValueError(f"{entry['id']}: invalid decoding_mode {mode!r}")

        unique_pids = []
        for p in parts:
            uid = normalize_participant_id(p, aliases)
            lane_sets[lane].add(uid)
            all_participants.add(uid)
            if uid not in unique_pids:
                unique_pids.append(uid)

        speech_abilities = []
        if any(p.endswith("\u25b2") for p in parts):
            speech_abilities.append("intact")
        if any(p.endswith("\u25b3") for p in parts):
            speech_abilities.append("paralysis")

        items.append((name_line, xm, shape, opn, metric_line, lane, mode))
        study_records.append({
            "idx": idx,
            "id": entry["id"],
            "citation": cit,
            "date": str(entry["date"]),
            "xm": xm,
            "name_line": name_line,
            "metric_line": metric_line,
            "descriptor": desc,
            "lane": lane,
            "electrode": elec_k,
            "shape": shape,
            "data_availability": avail_k,
            "opn": opn,
            "decoding_mode": mode,
            "task_instruction": task_k,
            "decoded_output": out_k,
            "speech_abilities": speech_abilities,
            "participants": list(parts),
            "unique_pids": unique_pids,
            "n_count": n_count,
            "metrics": dict(m_obj) if isinstance(m_obj, dict) else {"raw": str(m_obj)},
            "doi": str(entry.get("doi", "")).strip(),
            "data_doi": str(entry.get("data_doi", "")).strip(),
            "data_url": str(entry.get("data_url", "")).strip(),
            "url": str(entry.get("url", "")).strip(),
        })

    counts_summary = {
        "lane_totals": {
            lane: f"n = {len(lane_sets[lane])} unique participants"
            for lane in (LANE_SPEECH, LANE_TYPE, LANE_HAND)
        },
        "total_unique": len(all_participants),
        "home_summary": home_summary,
        "study_records": study_records,
        "metadata": meta,
    }
    return items, counts_summary


def check_figure_overlaps(fig: Figure) -> list[tuple[str, str, float]]:
    """Inspect all text and legend artists in ``fig`` for bounding-box collisions.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Rendered Matplotlib figure whose text and legend patches will be checked
        in display pixel coordinates.

    Returns
    -------
    list of tuple of (str, str, float)
        List of colliding pairs ``(label_1, label_2, overlap_percent_of_smaller)``
        whose pixel bounding boxes intersect by more than 1 px in both x and y.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = []
    for ax in fig.axes:
        for t in ax.texts:
            if t.get_text().strip():
                texts.append(t)
        if ax.axison:
            for t in ax.get_xticklabels() + ax.get_yticklabels():
                if t.get_text().strip():
                    texts.append(t)
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                if t.get_text().strip():
                    texts.append(t)
            for p in leg.get_patches():
                texts.append(p)
    bad = []
    for t1, t2 in itertools.combinations(texts, 2):
        try:
            b1, b2 = t1.get_window_extent(r), t2.get_window_extent(r)
        except Exception:
            continue
        if b1.overlaps(b2):
            ix = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
            iy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
            if ix > 1 and iy > 1:
                area = ix * iy
                small = min(b1.width * b1.height, b2.width * b2.height)
                s1 = getattr(t1, "get_text", lambda: "<patch>")()[:26].replace("\n", "/")
                s2 = getattr(t2, "get_text", lambda: "<patch>")()[:26].replace("\n", "/")
                bad.append((s1, s2, round(100 * area / max(small, 1), 1)))
    return bad


def render_timeline(
    yaml_path: str = "studies.yaml",
    out_stem: str = "communication_neuroprostheses_timeline",
    verify_overlaps: bool = True,
) -> tuple[int, dict[str, Any]]:
    """Render the static PNG/PDF timeline figures and the interactive HTML edition.

    Enforces all five verification gates (output-glyph/listener consistency,
    task-instruction diamond presence, green-on-track placement and marker spacing,
    equalized six-legend row gaps, and zero pairwise bounding-box overlaps) before
    writing ``<out_stem>.png``, ``<out_stem>.pdf``, and ``<out_stem>.html``.

    Parameters
    ----------
    yaml_path : str, default="studies.yaml"
        Path to the input YAML or JSON dataset.
    out_stem : str, default="communication_neuroprostheses_timeline"
        Output path stem (without extension) for generated ``.png``, ``.pdf``,
        and ``.html`` artifacts.
    verify_overlaps : bool, default=True
        Whether to run the pairwise pixel bounding-box overlap gate before saving.

    Returns
    -------
    n_cards : int
        Total number of study cards rendered.
    counts_summary : dict of str to Any
        Cohort accounting summary dictionary returned by :func:`load_studies`.

    Raises
    ------
    AssertionError
        If any visual invariant gate or card-packing constraint fails.
    RuntimeError
        If ``verify_overlaps`` is ``True`` and any two text/legend bounding boxes
        overlap by more than 1 px.
    """
    items, counts_summary = load_studies(yaml_path)

    # Gate 1: every card carries a pentagon on the metric line and none on line 1;
    # every "listener" card is voice-marked.
    _metrics = [it[4] for it in items]
    assert all(("\u2b1f" in t) or ("\u2b20" in t) for t in _metrics)
    assert not any(("\u2b1f" in it[0]) or ("\u2b20" in it[0]) for it in items)
    _voice = {it[0] for it in items if "\u2b1f" in it[4]}
    _listeners = {it[0] for it in items if "listener" in it[4]}
    assert _listeners <= _voice, (_listeners, _voice)

    # Gate 2: every citation carries a task-instruction diamond.
    _cites = [it[0] for it in items]
    assert all(("\u25c6" in t) or ("\u25c7" in t) for it in items for t in (it[0],))

    oc = {
        "open": OPEN_C[2],
        "partial": OPEN_C[1],
        "closed": OPEN_C[0],
        "unknown": OPEN_C["P"],
        "reanalysis": OPEN_C["R"],
    }

    fig, ax = plt.subplots(figsize=(27.0, 19.2))
    max_xm = max(it[1] for it in items)
    max_yr_tick = max(2028, int(max_xm) + 2)
    x0 = 2020.85
    x1 = max(2029.55, max_xm + 2.74)
    ax.set_xlim(x0, x1)
    ax.set_ylim(-4.62, 27.25)
    ax.axis("off")

    yax = -2.94
    vis_end = max(2027.10, (max_yr_tick - 1) + 0.10)
    ax.plot([x0, vis_end], [yax, yax], color="#3a3a3a", lw=1.2, zorder=3)
    for yr in range(2021, max_yr_tick):
        ax.plot([yr, yr], [yax, yax - 0.15], color="#3a3a3a", lw=1.2, zorder=3)
        ax.text(yr, -3.57, str(yr), ha="center", va="center",
                fontsize=10, color="#3a3a3a")

    TRACKS = {LANE_SPEECH: 18.28, LANE_TYPE: 8.98, LANE_HAND: -0.32}
    for lane, ty in TRACKS.items():
        ax.hlines(ty, x0, x1, color="#d3d3d3", lw=1.3, zorder=1)
        ax.text(x0 - 0.18, ty, lane.capitalize(), ha="right", va="center",
                fontsize=11.5, fontweight="bold", color="#404040")

    ROWS = {
        LANE_SPEECH: {"above": [19.83, 21.44, 23.05, 24.66, 26.27],
                      "below": [16.87, 15.32, 13.79, 12.24]},
        LANE_TYPE:   {"above": [10.12],      "below": [7.31]},
        LANE_HAND:   {"above": [0.82],       "below": [-1.46]},
    }
    GAP, PADX_L, PADX_R, MAXSHIFT = 0.13, x0 + 0.15, x1 - 0.12, 2.05

    SIDE, MARK = {}, {}
    counts = {lane: 0 for lane in ROWS}
    for lane in (LANE_HAND, LANE_TYPE, LANE_SPEECH):
        idxs = sorted((i for i, it in enumerate(items) if it[5] == lane),
                      key=lambda i: items[i][1])
        for i in idxs:
            if lane == LANE_SPEECH:
                side = "below" if counts[lane] % 2 == 0 else "above"
            elif lane == LANE_HAND and counts[lane] >= 2:
                side = "above" if counts[lane] % 2 == 0 else "below"
            else:
                side = "below"
            counts[lane] += 1
            SIDE[i] = side

    def resolve_marks(
        out: list[tuple[int, str, float, float, float, str, str, str, str]]
    ) -> dict[int, tuple[float, float]]:
        """Assign vertical track coordinates to every study marker and enforce Gate 3.

        Parameters
        ----------
        out : list of tuple
            Packed card tuples ``(i, name, xm, cx, y, s, shape, opn, metric)``.

        Returns
        -------
        dict of int to tuple of (float, float)
            Mapping from study index ``i`` to marker data coordinates ``(xm, my)``.
        """
        MARK.clear()
        for (i, name, xm, cx, y, s, shape, opn, metric) in out:
            lane = items[i][5]
            ty = TRACKS[lane]
            if opn == "open":
                MARK[i] = (xm, ty)
            else:
                lvl, sgn = 0.52, (1.0 if s == "above" else -1.0)
                while any(abs(xm - mx) < 0.045 and
                          abs(ty + sgn * lvl - my) < 1e-9
                          for (mx, my) in MARK.values()):
                    lvl += 0.52
                MARK[i] = (xm, ty + sgn * lvl)
        for i, it in enumerate(items):
            on_track = abs(MARK[i][1] - TRACKS[it[5]]) < 1e-9
            assert (it[3] == "open") == on_track, it[0]
        for lane in TRACKS:
            per_level: dict[float, list[tuple[float, float]]] = {}
            for i, it in enumerate(items):
                if it[5] != lane:
                    continue
                level = round(MARK[i][1], 6)
                for (mx, _my) in per_level.setdefault(level, []):
                    assert abs(mx - MARK[i][0]) >= 0.045, (it[0], lane)
                per_level[level].append(MARK[i])
        return MARK

    def pack(
        placed: list[tuple[float, float]], xm: float, hw: float, cap: float
    ) -> Optional[float]:
        """Find the leftmost collision-free horizontal center ``cx`` on a single row.

        Parameters
        ----------
        placed : list of tuple of (float, float)
            Already-placed ``(center_x, half_width)`` intervals on the candidate row.
        xm : float
            Study anchor time in decimal year units.
        hw : float
            Half-width of the candidate card in decimal year units.
        cap : float
            Maximum allowed rightward shift ``cx - xm`` in decimal year units.

        Returns
        -------
        float or None
            Packed horizontal center ``cx`` if feasible, otherwise ``None``.
        """
        cx = max(xm, PADX_L + hw)
        for pcx, phw in sorted(placed):
            if cx - hw < pcx + phw + GAP:
                cx = pcx + phw + hw + GAP
        if cx - xm > cap or cx + hw > PADX_R:
            return None
        return cx

    def layout(
        hws: list[float],
    ) -> list[tuple[int, str, float, float, float, str, str, str, str]]:
        """Pack all study cards into ``ROWS`` without horizontal overlap.

        Parameters
        ----------
        hws : list of float
            Card half-widths in decimal year units for each study in ``items``.

        Returns
        -------
        list of tuple
            Packed card descriptors ``(i, name, xm, cx, y, side, shape, opn, metric)``.
        """
        out = []
        placed = {lane: {y: [] for ys in ROWS[lane].values() for y in ys}
                  for lane in ROWS}
        LEG_X_END = x0 + 0.56 * (x1 - x0)
        TOP_ROW = ROWS[LANE_SPEECH]["above"][-1]
        placed[LANE_SPEECH][TOP_ROW].append(
            (0.5 * (x0 + LEG_X_END - GAP), 0.5 * (LEG_X_END - GAP - x0)))
        for i, (name, xm, shape, opn, metric, lane, _mode) in enumerate(items):
            got = None
            sides = ((SIDE[i],) if lane == LANE_SPEECH
                     else (SIDE[i], "above" if SIDE[i] == "below" else "below"))
            for cap in (0.45, MAXSHIFT):
                for s in sides:
                    for y in ROWS[lane][s]:
                        cx = pack(placed[lane][y], xm, hws[i], cap)
                        if cx is not None and not (
                                y == TOP_ROW and cx - hws[i] < LEG_X_END):
                            got = (cx, y, s)
                            break
                    if got:
                        break
                if got:
                    break
            assert got is not None, (
                f"Could not pack card {name!r}; increase x1 or add a label row in ROWS."
            )
            cx, y, s = got
            placed[lane][y].append((cx, hws[i]))
            out.append((i, name, xm, cx, y, s, shape, opn, metric))
        return out

    def draw(
        out: list[tuple[int, str, float, float, float, str, str, str, str]]
    ) -> tuple[list[Any], list[Any]]:
        """Render orthogonal leader lines and two-line study text cards onto ``ax``.

        Parameters
        ----------
        out : list of tuple
            Packed card descriptors produced by ``layout()``.

        Returns
        -------
        leads : list of matplotlib.lines.Line2D
            Leader-line artists added to ``ax``.
        texts : list of matplotlib.text.Text
            Two-line card text artists added to ``ax``.
        """
        leads, texts = [], []
        stem_x = {}
        by_lane_side = {}
        for entry in out:
            i, _n, xm, _cx, _y, side, *_ = entry
            by_lane_side.setdefault((items[i][5], side), []).append((xm, i))
        for (_lane, _side), lst in by_lane_side.items():
            lst.sort()
            k = 0
            while k < len(lst):
                grp = [lst[k]]
                while k + 1 < len(lst) and abs(lst[k + 1][0] - lst[k][0]) < 0.045:
                    k += 1
                    grp.append(lst[k])
                if len(grp) == 1:
                    stem_x[grp[0][1]] = grp[0][0]
                else:
                    n_g = len(grp)
                    for idx_g, (xm_g, i_g) in enumerate(grp):
                        off = (idx_g - 0.5 * (n_g - 1)) * 0.022
                        stem_x[i_g] = xm_g + off
                k += 1

        row_rails = {}
        for i, name, xm, cx, y, side, shape, opn, metric in out:
            lane = items[i][5]
            my = MARK[i][1]
            sgn = 1.0 if side == "above" else -1.0
            xs = stem_x.get(i, xm)
            y_start = my + sgn * 0.17
            if abs(cx - xs) > 0.02:
                x_lo, x_hi = min(xs, cx), max(xs, cx)
                prev_rails = row_rails.setdefault((lane, side, y), [])
                used_lvls = {lvl for (plo, phi, lvl) in prev_rails
                             if not (x_hi < plo - 0.04 or x_lo > phi + 0.04)}
                rail_lvl = 0
                while rail_lvl in used_lvls:
                    rail_lvl += 1
                prev_rails.append((x_lo, x_hi, rail_lvl))
                y_rail = y - sgn * (0.22 * rail_lvl)
                if abs(xs - xm) > 1e-4:
                    leads.append(ax.plot([xm, xs, xs], [y_start, y_start + sgn * 0.10, y_rail],
                                         color="#b8b8b8", lw=0.8, zorder=2)[0])
                else:
                    leads.append(ax.plot([xs, xs], [y_start, y_rail],
                                         color="#b8b8b8", lw=0.8, zorder=2)[0])
                leads.append(ax.plot([xs, cx], [y_rail, y_rail],
                                     color="#b8b8b8", lw=0.8, zorder=2)[0])
                if rail_lvl > 0:
                    leads.append(ax.plot([cx, cx], [y_rail, y],
                                         color="#b8b8b8", lw=0.8, zorder=2)[0])
            else:
                leads.append(ax.plot([xm, cx], [y_start, y],
                                     color="#b8b8b8", lw=0.8, zorder=2)[0])
            name_dy, met_dy = ((0.59, 0.22) if side == "above"
                               else (-0.22, -0.59))
            texts.append(ax.text(cx, y + name_dy, name, ha="center",
                                 va="center", fontsize=8, fontweight="bold",
                                 color="#1a1a1a", zorder=5))
            texts.append(ax.text(cx, y + met_dy, metric, ha="center",
                                 va="center", fontsize=7, color="#555555",
                                 zorder=5))
        return leads, texts

    # Pass 1: estimated widths -> Pass 2: measured renderer widths
    hws = [0.5 * 0.032 * max(len(n), len(m)) * 15.5 / 27.0 + 0.07
           for (n, _, _, _, m, _, _) in items]
    out1 = layout(hws)
    resolve_marks(out1)
    leads, texts = draw(out1)
    fig.canvas.draw()
    inv = ax.transData.inverted()
    renderer = fig.canvas.get_renderer()
    h_true = list(hws)
    for i in range(len(items)):
        wpx = max(texts[2 * i].get_window_extent(renderer).width,
                  texts[2 * i + 1].get_window_extent(renderer).width)
        p0 = inv.transform((0, 0))
        p1 = inv.transform((wpx, 0))
        h_true[i] = abs(p1[0] - p0[0]) / 2.0 + 0.06
    for a in leads + texts:
        a.remove()
    out = layout(h_true)
    resolve_marks(out)

    for i, (_, xm, shape, opn, _, lane, mode) in enumerate(items):
        if mode == "online":
            ax.plot([MARK[i][0]], [MARK[i][1]], marker=shape, ms=11.5,
                    markerfacecolor=oc[opn], markeredgecolor="white",
                    markeredgewidth=1.2, zorder=4, linestyle="")
        else:
            ax.plot([MARK[i][0]], [MARK[i][1]], marker=shape, ms=8.3,
                    markerfacecolor="white", markeredgecolor=oc[opn],
                    markeredgewidth=2.0, zorder=4, linestyle="")

    draw(out)

    h1 = [Line2D([], [], marker="o", ls="", color=oc[k],
                 markeredgecolor="white", markersize=10, label=lab)
          for k, lab in [("open", "open data"),
                         ("partial", "data on request"),
                         ("closed", "closed data"),
                         ("unknown", "preprint"),
                         ("reanalysis", "reanalysis on open data")]]
    h2 = [Line2D([], [], marker="o", ls="", markerfacecolor="#7F7F7F",
                 markeredgecolor="white", markeredgewidth=1.0,
                 markersize=10, label="online"),
          Line2D([], [], marker="o", ls="", markerfacecolor="white",
                 markeredgecolor="#7F7F7F", markeredgewidth=1.8,
                 markersize=7.2, label="offline")]
    h3 = [Line2D([], [], marker=m, ls="", color="#7F7F7F",
                 markeredgecolor="white", markersize=10, label=lab)
          for m, lab in [("o", "ECoG"), ("s", "Utah array"),
                         ("h", "thin-film threads"),
                         ("v", "microwire array")]]
    leg1 = ax.legend(handles=h1, loc="upper left", bbox_to_anchor=(0.002, 1.0),
                     title="data availability", fontsize=7.8, title_fontsize=8.2,
                     frameon=True, framealpha=0.95, edgecolor="#cccccc",
                     borderpad=0.7, labelspacing=0.45)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=h2, loc="upper left", bbox_to_anchor=(0.080, 1.0),
                     title="decoding mode", fontsize=7.8,
                     title_fontsize=8.2, frameon=True, framealpha=0.95,
                     edgecolor="#cccccc", borderpad=0.7, labelspacing=0.45)
    ax.add_artist(leg2)
    leg3 = ax.legend(handles=h3, loc="upper left", bbox_to_anchor=(0.155, 1.0),
                     title="electrode type", fontsize=7.8,
                     title_fontsize=8.2, frameon=True, framealpha=0.95,
                     edgecolor="#cccccc", borderpad=0.7, labelspacing=0.45)
    ax.add_artist(leg3)
    h4 = [Line2D([], [], marker="^", ls="", markerfacecolor="#1a1a1a",
                 markeredgecolor="white", markersize=8,
                 label="participants who can speak naturally"),
          Line2D([], [], marker="^", ls="", markerfacecolor="white",
                 markeredgecolor="#7F7F7F", markeredgewidth=0.7,
                 markersize=6.0,
                 label="participants who have vocal paralysis")]
    leg4 = ax.legend(handles=h4, loc="upper left", bbox_to_anchor=(0.231, 1.0),
                     title="speech ability", fontsize=7.8, title_fontsize=8.2,
                     frameon=True, framealpha=0.95, edgecolor="#cccccc",
                     borderpad=0.7, labelspacing=0.45)
    ax.add_artist(leg4)
    h5 = [Line2D([], [], marker="D", ls="", markerfacecolor="#1a1a1a",
                 markeredgecolor="#1a1a1a", markersize=3.65,
                 markeredgewidth=0.72,
                 label="inner speech"),
          Line2D([], [], marker="D", ls="", markerfacecolor="white",
                 markeredgecolor="#1a1a1a", markersize=3.65,
                 markeredgewidth=0.72,
                 label="overt/mouthing/intended")]
    leg5 = ax.legend(handles=h5, loc="upper left", bbox_to_anchor=(0.338, 1.0),
                     title="task instruction", fontsize=7.8, title_fontsize=8.2,
                     frameon=True, framealpha=0.95, edgecolor="#cccccc",
                     borderpad=0.7, labelspacing=0.45)
    ax.add_artist(leg5)
    h6 = [Line2D([], [], marker="p", ls="", markerfacecolor="#1a1a1a",
                 markeredgecolor="#1a1a1a", markersize=5.25,
                 markeredgewidth=0.72,
                 label="brain-to-voice"),
          Line2D([], [], marker="p", ls="", markerfacecolor="white",
                 markeredgecolor="#1a1a1a", markersize=5.25,
                 markeredgewidth=0.72,
                 label="brain-to-text")]
    leg6 = ax.legend(handles=h6, loc="upper left", bbox_to_anchor=(0.460, 1.0),
                     title="decoded output", fontsize=7.8,
                     title_fontsize=8.2, frameon=True, framealpha=0.95,
                     edgecolor="#cccccc", borderpad=0.7, labelspacing=0.45)

    ax.text(x1, -4.21,
            f"{counts_summary['total_unique']} unique participants in total; "
            f"{counts_summary['home_summary']}",
            ha="right", va="center", fontsize=8.2, style="italic",
            color="#3C5488", zorder=5)

    LANE_TOTALS = counts_summary["lane_totals"]
    for lane, ty in TRACKS.items():
        ax.text(x1 - 0.03, ty - 0.18, LANE_TOTALS[lane], ha="right",
                va="top", fontsize=10, fontweight="bold", color="#3C5488",
                zorder=5)

    fig.tight_layout()

    # Equal-gap redistribution across the 6 legend boxes
    fig.canvas.draw()
    legs = [leg1, leg2, leg3, leg4, leg5, leg6]
    ax_bb = ax.get_window_extent()
    ax_w = ax_bb.width
    bbs = [lg.get_window_extent() for lg in legs]
    row_start = (bbs[0].x0 - ax_bb.x0) / ax_w
    row_end = (bbs[-1].x1 - ax_bb.x0) / ax_w
    w_frac = [bb.width / ax_w for bb in bbs]
    gap = (row_end - row_start - sum(w_frac)) / (len(legs) - 1)
    assert gap > 0, (row_start, row_end, w_frac)
    xs, cur = [], row_start
    for w in w_frac:
        xs.append(cur)
        cur += w + gap
    for lg, x in zip(legs, xs):
        lg.set_bbox_to_anchor((x, 1.0))
    fig.canvas.draw()
    bbs = [lg.get_window_extent() for lg in legs]
    gaps_px = [bbs[i + 1].x0 - bbs[i].x1 for i in range(len(legs) - 1)]
    assert max(gaps_px) - min(gaps_px) <= 0.5, gaps_px
    span_px = bbs[-1].x1 - bbs[0].x0
    assert abs(span_px - (row_end - row_start) * ax_w) <= 0.5, span_px

    if verify_overlaps:
        bad = check_figure_overlaps(fig)
        if bad:
            plt.close(fig)
            msg = "\n".join(f"  {pct:5.1f}%: {a!r} <-> {b!r}" for a, b, pct in bad)
            raise RuntimeError(f"Overlap check failed:\n{msg}")

    # Capture exact geometry for the interactive HTML/SVG before closing `fig`
    fig_w_px, fig_h_px = fig.canvas.get_width_height()
    p_x0_y0 = ax.transData.transform((x0, -4.62))
    p_x1_y1 = ax.transData.transform((x1, 27.25))
    svg_geom = {
        "fig_w": round(fig_w_px * 0.5, 2),
        "fig_h": round(fig_h_px * 0.5, 2),
        "sx0": round(p_x0_y0[0] * 0.5, 3),
        "sy1": round((fig_h_px - p_x0_y0[1]) * 0.5, 3),
        "sx1": round(p_x1_y1[0] * 0.5, 3),
        "sy0": round((fig_h_px - p_x1_y1[1]) * 0.5, 3),
        "x0": x0,
        "x1": x1,
        "y0": -4.62,
        "y1": 27.25,
        "yax": yax,
        "vis_end": vis_end,
        "max_yr_tick": max_yr_tick,
        "tracks": TRACKS,
        "rows": ROWS,
        "gap": GAP,
        "padx_l": PADX_L,
        "padx_r": PADX_R,
        "maxshift": MAXSHIFT,
        "leg_x_end": x0 + 0.56 * (x1 - x0),
        "legend_boxes": [
            {
                "x": round(bb.x0 * 0.5, 2),
                "y": round((fig_h_px - bb.y1) * 0.5, 2),
                "w": round(bb.width * 0.5, 2),
                "h": round(bb.height * 0.5, 2),
            }
            for bb in bbs
        ],
    }

    out_by_idx = {entry[0]: entry for entry in out}
    study_records = counts_summary.get("study_records", [])
    for rec in study_records:
        idx = rec["idx"]
        _, _, _, cx, y, side, _, _, _ = out_by_idx[idx]
        rec["hw"] = round(float(h_true[idx]), 5)
        rec["static_cx"] = round(float(cx), 5)
        rec["static_y"] = round(float(y), 5)
        rec["static_side"] = side
        rec["static_mark_x"] = round(float(MARK[idx][0]), 5)
        rec["static_mark_y"] = round(float(MARK[idx][1]), 5)

    fig.savefig(out_stem + ".png", bbox_inches="tight")
    fig.savefig(out_stem + ".pdf", bbox_inches="tight")
    plt.close(fig)

    export_interactive_html(
        out_html=out_stem + ".html",
        studies=study_records,
        geom=svg_geom,
        counts_summary=counts_summary,
        verify_overlaps=verify_overlaps,
    )
    return len(items), counts_summary


FIG2_UPPER_DATA: dict[str, list[dict[str, Any]]] = {
    "panel_a_vocab_funnel": [
        {"id": "angrick2024", "label": "Angrick et al., 2024 (CC01)", "vocab_num": 6, "vocab_display": "6", "study_key": "Angrick et al., 2024", "is_open_vocab": False, "openness_code": 2},
        {"id": "wandelt2024", "label": "Wandelt et al., 2024 (P1/P2)", "vocab_num": 8, "vocab_display": "8", "study_key": "Wandelt et al., 2024", "is_open_vocab": False, "openness_code": 2},
        {"id": "brosler2026", "label": "Brosler et al., 2026 (B1r/B3/B6)", "vocab_num": 10, "vocab_display": "5\u201310", "study_key": "Brosler et al., 2026", "is_open_vocab": False, "openness_code": 1},
        {"id": "moses2021", "label": "Moses et al., 2021 (B1)", "vocab_num": 50, "vocab_display": "50", "study_key": "Moses et al., 2021", "is_open_vocab": False, "openness_code": 0},
        {"id": "silva2024b", "label": "Silva et al., 2024b (B1)", "vocab_num": 178, "vocab_display": "178", "study_key": "Silva et al., 2024b", "is_open_vocab": False, "openness_code": 1},
        {"id": "metzger2023", "label": "Metzger et al., 2023 (B3)", "vocab_num": 1024, "vocab_display": "1,024", "study_key": "Metzger et al., 2023", "is_open_vocab": False, "openness_code": 1},
        {"id": "littlejohn2025", "label": "Littlejohn et al., 2025 (B3)", "vocab_num": 1024, "vocab_display": "1,024", "study_key": "Littlejohn et al., 2025", "is_open_vocab": False, "openness_code": 1},
        {"id": "metzger2022", "label": "Metzger et al., 2022 (B1)", "vocab_num": 1152, "vocab_display": "1,152", "study_key": "Metzger et al., 2022", "is_open_vocab": False, "openness_code": 1},
        {"id": "feghhi2026", "label": "Feghhi et al., 2026 (T12/T15)", "vocab_num": 100000, "vocab_display": "100k", "study_key": "Feghhi et al., 2026", "is_open_vocab": False, "openness_code": "R"},
        {"id": "willett2023", "label": "Willett et al., 2023 (T12)", "vocab_num": 125000, "vocab_display": "125k", "study_key": "Willett et al., 2023", "is_open_vocab": False, "openness_code": 2},
        {"id": "card2024", "label": "Card et al., 2024 (T15)", "vocab_num": 125000, "vocab_display": "125k", "study_key": "Card et al., 2024", "is_open_vocab": False, "openness_code": 2},
        {"id": "feng2024", "label": "Feng et al., 2024 (T12)", "vocab_num": 125000, "vocab_display": "125k", "study_key": "Feng et al., 2024", "is_open_vocab": False, "openness_code": "R"},
        {"id": "kunz2025", "label": "Kunz et al., 2025 (T12/T15/T16)", "vocab_num": 125000, "vocab_display": "125k", "study_key": "Kunz et al., 2025", "is_open_vocab": False, "openness_code": 2},
        {"id": "li2025", "label": "Li et al., 2025 (T12)", "vocab_num": 125000, "vocab_display": "125k", "study_key": "Li et al., 2025", "is_open_vocab": False, "openness_code": "R"},
        {"id": "feghhi2025", "label": "Feghhi et al., 2025 (T12)", "vocab_num": 125000, "vocab_display": "125k", "study_key": "Feghhi et al., 2025", "is_open_vocab": False, "openness_code": "R"},
        {"id": "levin2026_speech", "label": "Levin et al., 2026 (T12/T15/T16)", "vocab_num": 125000, "vocab_display": "125k", "study_key": "Levin et al., 2026", "is_open_vocab": False, "openness_code": "P"},
        {"id": "yzhang2026_bit", "label": "Y. Zhang et al., 2026 (T12/T15)", "vocab_num": 125000, "vocab_display": "125k", "study_key": "Y. Zhang et al., 2026", "is_open_vocab": False, "openness_code": "R"},
        {"id": "nason2026", "label": "Nason-Tomaszewski\net al., 2026 (T16)", "vocab_num": 125000, "vocab_display": "125k", "study_key": "Nason-Tomaszewski et al., 2026", "is_open_vocab": False, "openness_code": "P"},
        {"id": "boccato2026", "label": "Boccato et al., 2026\n(T12/T15/T16/T17)", "vocab_num": 125000, "vocab_display": "125k", "study_key": "Boccato et al., 2026", "is_open_vocab": False, "openness_code": "R"},
        {"id": "jude2026a", "label": "Jude et al., 2026a (T17/T18)", "vocab_num": 125000, "vocab_display": "125k", "study_key": "Jude et al., 2026a", "is_open_vocab": False, "openness_code": 2},
        {"id": "jude2026b", "label": "Jude et al., 2026b (T17)", "vocab_num": 125000, "vocab_display": "125k", "study_key": "Jude et al., 2026b", "is_open_vocab": False, "openness_code": 2},
        {"id": "yoon2026", "label": "Yoon et al., 2026 (T12)", "vocab_num": 125000, "vocab_display": "125k", "study_key": "Yoon et al., 2026", "is_open_vocab": False, "openness_code": "P"},
        {"id": "card2026", "label": "Card et al., 2026 (T15)", "vocab_num": 125000, "vocab_display": "125k", "study_key": "Card et al., 2026", "is_open_vocab": False, "openness_code": 0},
        {"id": "fogg2026", "label": "Fogg et al., 2026\n(T12/T15/T16/T17/T21/T22)", "vocab_num": 125000, "vocab_display": "125k", "study_key": "Fogg et al., 2026", "is_open_vocab": False, "openness_code": "P"},
        {"id": "zzhang2026_align", "label": "Z. Zhang et al., 2026 (T12/T15)", "vocab_num": 125000, "vocab_display": "125k", "study_key": "Z. Zhang et al., 2026", "is_open_vocab": False, "openness_code": "R"},
        {"id": "pouratian2026", "label": "Pouratian et al., 2026 (V1/V2/V3)", "vocab_num": 125000, "vocab_display": "125k", "study_key": "Pouratian et al., 2026", "is_open_vocab": False, "openness_code": 0},
        {"id": "fan2023", "label": "Fan et al., 2023 (T5)", "vocab_num": 130000, "vocab_display": "130k", "study_key": "Fan et al., 2023", "is_open_vocab": False, "openness_code": 2},
        {"id": "karpowicz2024", "label": "Karpowicz et al., 2024 (T5)", "vocab_num": 130000, "vocab_display": "130k", "study_key": "Karpowicz et al., 2024", "is_open_vocab": False, "openness_code": "R"},
        {"id": "willett2021", "label": "Willett et al., 2021 (T5)", "vocab_num": 300000, "vocab_display": "open", "study_key": "Willett et al., 2021", "is_open_vocab": True, "openness_code": 2},
        {"id": "wairagkar2025", "label": "Wairagkar et al., 2025 (T15)", "vocab_num": 300000, "vocab_display": "open", "study_key": "Wairagkar et al., 2025", "is_open_vocab": True, "openness_code": 2},
        {"id": "levin2026_handwriting", "label": "Levin et al., 2026 (T5/T12/T18)", "vocab_num": 300000, "vocab_display": "open", "study_key": "Levin et al., 2026", "is_open_vocab": True, "openness_code": "P"},
        {"id": "wairagkar2026", "label": "Wairagkar et al., 2026 (T15)", "vocab_num": 300000, "vocab_display": "open", "study_key": "Wairagkar et al., 2026", "is_open_vocab": True, "openness_code": "P"},
    ],
    "panel_b_dataset_size": [
        {"id": "willett2021", "label": "Willett et al., 2021 (T5)", "released_sentences": 1000, "sentences_display": "1,000", "study_key": "Willett et al., 2021", "span_note": "10 d/sess\n95 blks\nspan 253 d", "openness_code": 2},
        {"id": "willett2023", "label": "Willett et al., 2023 (T12)", "released_sentences": 9680, "sentences_display": "9,680", "study_key": "Willett et al., 2023", "span_note": "24 d/sess\n275 blks\nspan 120 d", "openness_code": 2},
        {"id": "fan2023", "label": "Fan et al., 2023 (T5)", "released_sentences": 1315, "sentences_display": "1,315", "study_key": "Fan et al., 2023", "span_note": "21 d/sess\n105 blks\nspan 287 d", "openness_code": 2},
        {"id": "card2024", "label": "Card et al., 2024 (T15)", "released_sentences": 9498, "sentences_display": "9,498", "study_key": "Card et al., 2024", "span_note": "45 d/sess\n265 blks\nspan 612 d", "openness_code": 2},
        {"id": "wairagkar2025", "label": "Wairagkar et al., 2025 (T15)", "released_sentences": 8489, "sentences_display": "8,489", "study_key": "Wairagkar et al., 2025", "span_note": "37 d/sess\nblks n/a\nspan 171 d", "openness_code": 2},
        {"id": "kunz2025", "label": "Kunz et al., 2025 (T12)", "released_sentences": 500, "sentences_display": "500", "study_key": "Kunz et al., 2025", "span_note": "1 d/sess\n12 blks\nspan 1 d", "openness_code": 2},
        {"id": "kunz2025", "label": "Kunz et al., 2025 (T15)", "released_sentences": 720, "sentences_display": "720", "study_key": "Kunz et al., 2025", "span_note": "3 d/sess\n18 blks\nspan 309 d", "openness_code": 2},
        {"id": "kunz2025", "label": "Kunz et al., 2025 (T16)", "released_sentences": 450, "sentences_display": "450", "study_key": "Kunz et al., 2025", "span_note": "2 d/sess\n14 blks\nspan 234 d", "openness_code": 2},
        {"id": "jude2026a", "label": "Jude et al., 2026a (T17)", "released_sentences": 668, "sentences_display": "668", "study_key": "Jude et al., 2026a", "span_note": "8 d/sess\n46 blks\nspan 89 d", "openness_code": 2},
        {"id": "jude2026a", "label": "Jude et al., 2026a (T18)", "released_sentences": 1033, "sentences_display": "1,033", "study_key": "Jude et al., 2026a", "span_note": "9 d/sess\n98 blks\nspan 119 d", "openness_code": 2},
        {"id": "jude2026b", "label": "Jude et al., 2026b (T17)", "released_sentences": 3674, "sentences_display": "3,674", "study_key": "Jude et al., 2026b", "span_note": "13 d/sess\n96 blks\nspan 141 d", "openness_code": 2},
    ],
    "panel_cd_online_wer_speed": [
        {"id": "willett2021", "label": "Willett et al., 2021 (T5)", "wer_mean": 25.1, "wer_sd": 8.6, "speed_wpm_mean": 15.4, "speed_wpm_sd": 2.2, "study_key": "Willett et al., 2021", "vocab_tag": "open", "wer_caliber": "mean\u00b1SD\u2020\n5 of 10 d", "speed_caliber": "mean\u00b1SD\u2020\n5 of 10 d", "openness_code": 2},
        {"id": "willett2023", "label": "Willett et al., 2023 (T12)", "wer_mean": 23.9, "wer_sd": 1.8, "speed_wpm_mean": 65.3, "speed_wpm_sd": 0.9, "study_key": "Willett et al., 2023", "vocab_tag": "125k", "wer_caliber": "mean\u00b1SD\u2021\n5 of 24 d", "speed_caliber": "mean\u00b1SD\u2021\n5 of 24 d", "openness_code": 2},
        {"id": "fan2023", "label": "Fan et al., 2023 (T5)", "wer_mean": 4.0, "wer_sd": 1.2, "speed_wpm_mean": 14.6, "speed_wpm_sd": 2.2, "study_key": "Fan et al., 2023", "vocab_tag": "130k", "wer_caliber": "mean\u00b1SD\u2020\n10 of 21 d", "speed_caliber": "mean\u00b1SD\n15 eval d\n(10 released)", "openness_code": 2},
        {"id": "card2024", "label": "Card et al., 2024 (T15)", "wer_mean": 4.9, "wer_sd": 3.3, "speed_wpm_mean": 32.5, "speed_wpm_sd": 2.7, "study_key": "Card et al., 2024", "vocab_tag": "125k", "wer_caliber": "mean\u00b1SD\n14 of 45 d", "speed_caliber": "mean\u00b1SD\n14 of 45 d", "openness_code": 2},
        {"id": "wairagkar2025", "label": "Wairagkar et al., 2025 (T15)", "wer_mean": 43.8, "wer_sd": None, "speed_wpm_mean": None, "speed_wpm_sd": None, "study_key": "Wairagkar et al., 2025", "vocab_tag": "open", "wer_caliber": "listener median\n8 of 37 d", "speed_caliber": "n/r", "openness_code": 2},
        {"id": "kunz2025", "label": "Kunz et al., 2025 (T12)", "wer_mean": 23.7, "wer_sd": 0.0, "speed_wpm_mean": None, "speed_wpm_sd": None, "study_key": "Kunz et al., 2025", "vocab_tag": "50-w", "wer_caliber": "mean\u00b1SD\u2020\n1 of 1 d", "speed_caliber": "n/r", "openness_code": 2},
        {"id": "kunz2025", "label": "Kunz et al., 2025 (T15)", "wer_mean": 51.3, "wer_sd": 33.1, "speed_wpm_mean": None, "speed_wpm_sd": None, "study_key": "Kunz et al., 2025", "vocab_tag": "125k", "wer_caliber": "mean\u00b1SD\u2020\n2 of 3 d", "speed_caliber": "n/r", "openness_code": 2},
        {"id": "kunz2025", "label": "Kunz et al., 2025 (T16)", "wer_mean": 46.6, "wer_sd": 0.0, "speed_wpm_mean": None, "speed_wpm_sd": None, "study_key": "Kunz et al., 2025", "vocab_tag": "125k", "wer_caliber": "mean\u00b1SD\u2020\n1 of 2 d", "speed_caliber": "n/r", "openness_code": 2},
        {"id": "jude2026a", "label": "Jude et al., 2026a (T17)", "wer_mean": 12.5, "wer_sd": 2.5, "speed_wpm_mean": 6.8, "speed_wpm_sd": 1.8, "study_key": "Jude et al., 2026a", "vocab_tag": "125k", "wer_caliber": "mean\u00b1SD\u2020\n6 of 8 d", "speed_caliber": "mean\u00b1SD\u2020\n6 of 8 d", "openness_code": 2},
        {"id": "jude2026a", "label": "Jude et al., 2026a (T18)", "wer_mean": 3.4, "wer_sd": 0.8, "speed_wpm_mean": 14.0, "speed_wpm_sd": 4.5, "study_key": "Jude et al., 2026a", "vocab_tag": "125k", "wer_caliber": "mean\u00b1SD\u2020\n7 of 9 d", "speed_caliber": "mean\u00b1SD\u2020\n7 of 9 d", "openness_code": 2},
        {"id": "jude2026b", "label": "Jude et al., 2026b (T17)", "wer_mean": 66.5, "wer_sd": 18.1, "speed_wpm_mean": 34.1, "speed_wpm_sd": 7.7, "study_key": "Jude et al., 2026b", "vocab_tag": "125k", "wer_caliber": "mean\u00b1SD\u2020\n7 of 13 d", "speed_caliber": "mean\u00b1SD\u2020\n7 of 13 d", "openness_code": 2},
    ],
}


def _build_fig2_upper_svg(
    studies: list[dict[str, Any]], verify_overlaps: bool = True
) -> tuple[str, int, int]:
    """Render Figure 2's upper half (Panels a-d + funnel brace) as interactive SVG.

    Constructs the exact Matplotlib layout for Panels (a)-(d), runs the pairwise
    text bounding-box overlap check when ``verify_overlaps`` is True, and emits
    semantic, clickable SVG markup linked to each study's timeline ``id``.

    Parameters
    ----------
    studies : list of dict of str to Any
        Loaded study records used to validate every bar's ``id``.
    verify_overlaps : bool, default=True
        Whether to run the bounding-box collision check on the Matplotlib figure.

    Returns
    -------
    svg_inner : str
        Inner SVG markup for ``<svg id="f2-svg">``.
    vw : int
        SVG viewBox width.
    vh : int
        SVG viewBox height.
    """
    import html as html_lib
    import matplotlib.patches as mpatches
    from matplotlib.colors import to_rgb
    from matplotlib.path import Path as MplPath
    from matplotlib.text import Text

    valid_ids = {rec["id"] for rec in studies}
    rows_a = FIG2_UPPER_DATA["panel_a_vocab_funnel"]
    rows_b = FIG2_UPPER_DATA["panel_b_dataset_size"]
    rows_cd = FIG2_UPPER_DATA["panel_cd_online_wer_speed"]
    for grp in (rows_a, rows_b, rows_cd):
        for r in grp:
            assert r["id"] in valid_ids, f"Unknown study id {r['id']!r} in FIG2_UPPER_DATA"

    open_how = {
        "Willett et al., 2021": "brain-to-text\nwithout\nlanguage model",
        "Wairagkar et al., 2025": "brain-to-voice",
        "Levin et al., 2026": "brain-to-text\nwithout\nlanguage model",
        "Wairagkar et al., 2026": "brain-to-voice",
    }

    fig_h = 14.05
    fig = plt.figure(figsize=(22.4, fig_h), dpi=200)
    outer = fig.add_gridspec(3, 1, height_ratios=(1.02, 0.08, 0.96), hspace=0.830447)
    fig.subplots_adjust(left=0.045, right=0.985, top=1.0 - 0.92 / fig_h, bottom=2.95 / fig_h)

    ax1 = fig.add_subplot(outer[0])
    for x, r in enumerate(rows_a):
        v = r["vocab_num"]
        end = r["vocab_display"]
        is_open = r["is_open_vocab"]
        state_c = OPEN_C[r["openness_code"]]
        if is_open:
            ax1.bar(x, v, width=0.76, color=(*to_rgb(state_c), 0.18),
                    edgecolor=state_c, linewidth=0.8, linestyle=(0, (3, 2)))
        else:
            ax1.bar(x, v, width=0.76, color=(*to_rgb(state_c), 0.35),
                    edgecolor=state_c, linewidth=0.8)
        ax1.text(x, v * 1.15, end, ha="center", va="bottom", fontsize=7.6)
        how = open_how.get(r["study_key"]) if is_open else None
        if how is not None:
            ax1.text(x, v * 0.85, how, ha="center", va="top",
                     fontsize=4.6, color="#555555", zorder=3)
    for anchor_v, anchor_lab, above in (
            (11100, "11,100 word families", False),
            (42000, "42,000 lemmas", False),
            (71400, "71,400 word forms", True)):
        ax1.axhline(anchor_v, color="#7F7F7F", linewidth=0.9, linestyle=(0, (4, 2)))
        if above:
            ax1.text(-0.55, anchor_v * 1.12, anchor_lab + " (Brysbaert et al., 2016)",
                     ha="left", va="bottom", fontsize=7.0, color="#555555")
        else:
            ax1.text(-0.55, anchor_v * 0.72, anchor_lab + " (Brysbaert et al., 2016)",
                     ha="left", va="top", fontsize=7.0, color="#555555")
    ax1.set_yscale("log")
    ax1.set_ylim(2.5, 1.5e6)
    ax1.set_yticks([10 ** k for k in range(0, 7)])
    ax1.set_yticklabels(["1", "10", "100", "1,000", "10,000", "100,000", "1,000,000"])
    ax1.yaxis.set_minor_locator(
        matplotlib.ticker.FixedLocator(
            [m * 10 ** k for k in range(0, 6) for m in range(2, 10) if m * 10 ** k >= 2.5]
        )
    )
    ax1.set_xticks(range(len(rows_a)))
    ax1.set_xticklabels([r["label"] for r in rows_a], fontsize=6.7, rotation=75,
                        ha="right", rotation_mode="anchor")
    ax1.set_xlim(-0.7, len(rows_a) - 0.3)
    ax1.set_ylabel("vocabulary size")
    ax1.set_title(
        f"(a) Maximum vocabulary size used in the {len(rows_a)} English communication-neuroprostheses studies",
        fontsize=10, fontweight="bold"
    )

    STAR_IDS = {"fan2023", "card2024", "jude2026a"}
    gs_r2 = outer[2].subgridspec(1, 5, width_ratios=(1.176, 0.12, 1.176, 0.12, 1.176), wspace=0.0)
    ax_b = fig.add_subplot(gs_r2[0, 0])
    ax2 = fig.add_subplot(gs_r2[0, 2])
    ax3 = fig.add_subplot(gs_r2[0, 4])

    for x, r in enumerate(rows_b):
        n_sent = r["released_sentences"]
        state_c = OPEN_C[r["openness_code"]]
        ax_b.bar(x, n_sent, width=0.76, color=(*to_rgb(state_c), 0.35),
                 edgecolor=state_c, linewidth=0.8)
        ax_b.text(x, n_sent + 220, r["sentences_display"], ha="center", va="bottom", fontsize=7.3)
        ax_b.text(x, n_sent + 1150, r["span_note"], ha="center", va="bottom",
                  fontsize=4.6, color="#555555", linespacing="normal")
    ax_b.set_ylim(0, 14500)
    ax_b.set_yticks(range(0, 12001, 2000))
    ax_b.set_yticklabels([f"{v:,}" for v in range(0, 12001, 2000)])
    ax_b.set_xticks(range(len(rows_b)))
    ax_b.set_xticklabels([("\u2605 " + r["label"]) if r["id"] in STAR_IDS else r["label"] for r in rows_b],
                         fontsize=6.4, rotation=75, ha="right", rotation_mode="anchor")
    ax_b.set_xlim(-0.75, len(rows_b) - 0.25)
    ax_b.set_ylabel("released sentences")
    ax_b.set_title("(b) Open-sourced dataset size\n", fontsize=9.8, fontweight="bold")

    c_badges = []
    for x, r in enumerate(rows_cd):
        wmean = r["wer_mean"]
        wsd = r["wer_sd"]
        state_c = OPEN_C[r["openness_code"]]
        ax2.bar(x, wmean, width=0.76, color=(*to_rgb(state_c), 0.35),
                edgecolor=state_c, linewidth=0.8)
        top = wmean + wsd if wsd is not None else wmean
        if wsd is not None:
            ax2.errorbar(x, wmean, yerr=wsd, color="#333333", capsize=2.0,
                         elinewidth=1.0, capthick=1.0, zorder=4)
        note = f"{wmean:.1f}\u00b1{wsd:.1f}" if wsd is not None else f"{wmean:.1f}"
        ax2.text(x, top + 1.6, note, ha="center", va="bottom", fontsize=6.6)
        ax2.text(x, top + 7.0, f"vocab {r['vocab_tag']}\n{r['wer_caliber']}",
                 ha="center", va="bottom", fontsize=4.6, color="#555555", linespacing="normal")
        if r["id"] in STAR_IDS:
            ax2.text(x, top + 18.8, "\u2605", ha="center", va="bottom", fontsize=7.8, fontweight="bold")
    for asr_v in (2.5, 5.8, 12.8):
        ax2.axhline(asr_v, color="#7F7F7F", linewidth=0.9, linestyle=(0, (4, 2)))
    for asr_v, asr_lab, ly, lx_lead in (
            (2.5, "2.5% LibriSpeech (Radford et al., 2022)", 62.6, 4.55),
            (5.8, "5.8% Switchboard (Xiong et al., 2017)", 68.1, 3.44),
            (12.8, "12.8% 14-dataset average (Radford et al., 2022)", 73.6, 2.50)):
        ax2.plot([lx_lead, lx_lead], [asr_v, ly - 0.2], color="#999999", lw=0.55, zorder=2)
        t_obj = ax2.text(5.50, ly, asr_lab, ha="right", va="bottom", fontsize=5.6,
                         color="#666666", zorder=3,
                         bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.88))
        c_badges.append((asr_v, asr_lab, ly, lx_lead, t_obj))
    ax2.set_ylim(0, 125)
    ax2.set_yticks(range(0, 101, 25))
    ax2.set_xticks(range(len(rows_cd)))
    ax2.set_xticklabels([("\u2605 " + r["label"]) if r["id"] in STAR_IDS else r["label"] for r in rows_cd],
                        fontsize=6.4, rotation=75, ha="right", rotation_mode="anchor")
    ax2.set_xlim(-0.75, len(rows_cd) - 0.25)
    ax2.set_ylabel("word error rate (%)")
    ax2.set_title(
        "(c) Word error rate (online)\n"
        "\u2605 3 studies satisfy: Open Data, Vocab > 71.4k, WER < 12.8%",
        fontsize=9.8, fontweight="bold"
    )
    dagger_note = (
        "\u2020 recomputed from released data (decode outputs / trial timing / Source Data files)\n"
        "\u2021 digitized from the paper's figure markers\n"
        "error bar is calculated across days"
    )
    ax2.text(0.98, 0.985, dagger_note, transform=ax2.transAxes, ha="right", va="top",
             fontsize=6.0, color="#555555", linespacing="normal")

    d_badges = []
    for x, r in enumerate(rows_cd):
        spd = r["speed_wpm_mean"]
        spdsd = r["speed_wpm_sd"]
        state_c = OPEN_C[r["openness_code"]]
        if spd is None:
            ax3.text(x, 1.2, "n/r", ha="center", va="bottom", fontsize=5.6, color="#777777")
            continue
        ax3.bar(x, spd, width=0.76, color=(*to_rgb(state_c), 0.35),
                edgecolor=state_c, linewidth=0.8)
        top = spd + spdsd if spdsd is not None else spd
        if spdsd is not None:
            ax3.errorbar(x, spd, yerr=spdsd, color="#333333", capsize=2.0,
                         elinewidth=1.0, capthick=1.0, zorder=4)
        note = f"{spd:.1f}\u00b1{spdsd:.1f}" if spdsd is not None else f"{spd:.1f}"
        ax3.text(x, top * 1.06 + 2.0, note, ha="center", va="bottom", fontsize=6.8)
        ax3.text(x, max(top * 1.31, top * 1.06 + 8.0) + 2.0,
                 f"vocab {r['vocab_tag']}\n{r['speed_caliber']}",
                 ha="center", va="bottom", fontsize=4.6, color="#555555", linespacing="normal")
    for spd_v in (160, 51.6, 36.2, 10, 8):
        ax3.axhline(spd_v, color="#7F7F7F", linewidth=0.9, linestyle=(0, (4, 2)))
    LX = 3.92
    for spd_v, spd_lab in (
            (160, "160 wpm natural conversation (Card et al., 2024)"),
            (51.6, "51.6 wpm keyboard typing (Dhakal et al., 2018)"),
            (36.2, "36.2 wpm smartphone typing (Palin et al., 2019)"),
            (10, "5\u201310 wpm eye typing (Hamid & Kristensson, 2024)")):
        t_obj = ax3.text(LX, spd_v + 0.6, spd_lab, ha="left", va="bottom", fontsize=5.6,
                         color="#666666", zorder=3,
                         bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.88))
        d_badges.append((spd_v + 0.6, spd_lab, t_obj))
    ax3.plot([5.5, 5.5], [8, 19.8], color="#999999", lw=0.55, zorder=2)
    t_obj_8 = ax3.text(LX, 20.0, "8 wpm point-and-click BCI (Willett et al., 2021)",
                       ha="left", va="bottom", fontsize=5.6, color="#666666", zorder=3,
                       bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.88))
    d_badges.append((20.0, "8 wpm point-and-click BCI (Willett et al., 2021)", t_obj_8))
    ax3.set_ylim(0, 200)
    ax3.set_yticks(range(0, 201, 50))
    ax3.set_xticks(range(len(rows_cd)))
    ax3.set_xticklabels([("\u2605 " + r["label"]) if r["id"] in STAR_IDS else r["label"] for r in rows_cd],
                        fontsize=6.4, rotation=75, ha="right", rotation_mode="anchor")
    ax3.set_xlim(-0.75, len(rows_cd) - 0.25)
    ax3.set_ylabel("words per minute")
    ax3.set_title("(d) Communication speed (online)\nn/r: speed not reported",
                  fontsize=9.8, fontweight="bold")
    ax3.text(0.98, 0.985, dagger_note, transform=ax3.transAxes, ha="right", va="top",
             fontsize=6.0, color="#555555", linespacing="normal")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv_fig = fig.transFigure.inverted()
    selected_idx = [
        i for i, r in enumerate(rows_a)
        if r["vocab_num"] > 71400 and r["openness_code"] == 2
    ]
    xtick_objs = ax1.get_xticklabels()
    drop_points = []
    min_lbl_y0 = 1.0
    for i, lbl in enumerate(xtick_objs):
        bb = inv_fig.transform_bbox(lbl.get_window_extent(renderer))
        if bb.y0 < min_lbl_y0:
            min_lbl_y0 = bb.y0
        if i in selected_idx:
            x_anchor = 0.5 * (bb.x0 + bb.x1)
            bar_x = inv_fig.transform(ax1.transData.transform((i, 2.5)))[0]
            x_pt = 0.65 * bar_x + 0.35 * x_anchor
            drop_points.append((rows_a[i]["id"], x_pt, bb.y0 - 0.0045))

    title_y1 = max(
        inv_fig.transform_bbox(ax.title.get_window_extent(renderer)).y1
        for ax in (ax_b, ax2, ax3)
    )
    brace_x0 = ax_b.get_position().x0
    brace_x1 = ax3.get_position().x1
    brace_xm = 0.5 * (brace_x0 + brace_x1)

    corridor = min_lbl_y0 - title_y1
    y_bus = min_lbl_y0 - 0.28 * corridor
    y_apex = title_y1 + 0.42 * corridor
    y_arm = title_y1 + 0.22 * corridor
    y_tip = title_y1 + 0.06 * corridor
    guide_c = OPEN_C[2]
    fig.text(
        brace_xm - 0.008, 0.5 * (y_bus + y_arm) + 0.001,
        "Filtered to 8 open-data studies exceeding native-speaker vocabulary (>71,400 word forms)",
        ha="right", va="center", fontsize=7.4, fontweight="bold", color=guide_c
    )

    if verify_overlaps:
        fig.canvas.draw()
        r_chk = fig.canvas.get_renderer()
        for ax in fig.axes:
            items_t = []
            for t in ax.findobj(Text):
                if not t.get_visible() or not t.get_text().strip():
                    continue
                bb = t.get_window_extent(r_chk)
                if bb.width <= 1 or bb.height <= 1:
                    continue
                items_t.append((t.get_text().strip().replace("\n", " | "), bb))
            for i in range(len(items_t)):
                s1, b1 = items_t[i]
                for j in range(i + 1, len(items_t)):
                    s2, b2 = items_t[j]
                    dx = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
                    dy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
                    if dx > 1.0 and dy > 1.0:
                        plt.close(fig)
                        raise RuntimeError(f"Figure 2 upper-half overlap: {s1!r} <-> {s2!r}")

    # Convert Matplotlib pixel coordinates to 0.5-scale SVG coordinates (100 units/in)
    w_px, h_px = fig.canvas.get_width_height()
    vw, vh = int(round(w_px * 0.5)), int(round(h_px * 0.5))
    PT = 1.3889
    SW = lambda pt: round(pt * PT, 2)
    SX = lambda px: px * 0.5
    SY = lambda py: (h_px - py) * 0.5
    esc = html_lib.escape

    def d2svg(ax, dx, dy):
        px, py = ax.transData.transform((dx, dy))
        return SX(px), SY(py)

    def f2svg(fx, fy):
        px, py = fig.transFigure.transform((fx, fy))
        return SX(px), SY(py)

    def multiline_svg(
        sx: float, sy: float, text: str, fs_pt: float, color: str,
        anchor: str = "middle", va: str = "bottom", weight: str = "normal",
        cls: str = ""
    ) -> str:
        lines = text.split("\n")
        fs = fs_pt * PT
        lh = fs * 1.20
        cls_attr = f' class="{cls}"' if cls else ""
        w_attr = f' font-weight="{weight}"' if weight != "normal" else ""
        if len(lines) == 1:
            if va == "bottom":
                yy = sy - 0.12 * fs
                dom = "alphabetic"
            elif va == "top":
                yy = sy + 0.82 * fs
                dom = "alphabetic"
            else:
                yy = sy
                dom = "central"
            return (f'<text{cls_attr} x="{sx:.1f}" y="{yy:.1f}" font-size="{fs:.2f}" '
                    f'fill="{color}"{w_attr} text-anchor="{anchor}" '
                    f'dominant-baseline="{dom}">{esc(lines[0])}</text>')
        out_t = [f'<text{cls_attr} x="{sx:.1f}" font-size="{fs:.2f}" fill="{color}"'
                 f'{w_attr} text-anchor="{anchor}">']
        n_l = len(lines)
        for k, line in enumerate(lines):
            if va == "bottom":
                yy = sy - (n_l - 1 - k) * lh - 0.14 * fs
            elif va == "top":
                yy = sy + k * lh + 0.82 * fs
            else:
                yy = sy + (k - 0.5 * (n_l - 1)) * lh + 0.32 * fs
            out_t.append(f'<tspan x="{sx:.1f}" y="{yy:.1f}">{esc(line)}</tspan>')
        out_t.append("</text>")
        return "".join(out_t)

    STAR_IDS = {"fan2023", "card2024", "jude2026a"}

    def rot_xtick_svg(tx: float, ty: float, label: str, fs_pt: float,
                      bold: bool = False, color: str = "#1f2937") -> str:
        lines = label.split("\n")
        fs = fs_pt * PT
        lh = fs * 1.15
        w_attr = ' font-weight="bold"' if bold else ""
        if len(lines) == 1:
            return (f'<g transform="translate({tx:.1f},{ty:.1f}) rotate(-75)">'
                    f'<text class="f2-lbl" x="0" y="{0.34 * fs:.1f}" font-size="{fs:.2f}" '
                    f'fill="{color}"{w_attr} text-anchor="end">{esc(lines[0])}</text></g>')
        spans = []
        for k, line in enumerate(lines):
            yy = (k - 0.35 * (len(lines) - 1)) * lh + 0.34 * fs
            spans.append(f'<tspan x="0" y="{yy:.1f}">{esc(line)}</tspan>')
        return (f'<g transform="translate({tx:.1f},{ty:.1f}) rotate(-75)">'
                f'<text class="f2-lbl" font-size="{fs:.2f}" fill="{color}"{w_attr} '
                f'text-anchor="end">{"".join(spans)}</text></g>')

    def draw_axes_chrome(ax, yticks, yticklabels, ylabel_str, title_str,
                         title_fs=9.8, minor_yticks=None, sub_fs=None, sub_color="#111827"):
        bb = ax.get_window_extent(renderer)
        ax_x0, ax_x1 = SX(bb.x0), SX(bb.x1)
        ax_y0, ax_y1 = SY(bb.y1), SY(bb.y0)  # ax_y0 is top, ax_y1 is bottom
        c = [
            f'<line x1="{ax_x0:.1f}" y1="{ax_y0:.1f}" x2="{ax_x0:.1f}" y2="{ax_y1:.1f}" '
            f'stroke="#1f2937" stroke-width="{SW(0.8)}"/>',
            f'<line x1="{ax_x0:.1f}" y1="{ax_y1:.1f}" x2="{ax_x1:.1f}" y2="{ax_y1:.1f}" '
            f'stroke="#1f2937" stroke-width="{SW(0.8)}"/>',
        ]
        for yv, ylab in zip(yticks, yticklabels):
            _, sy = d2svg(ax, 0, yv)
            c.append(f'<line x1="{ax_x0 - 3.5:.1f}" y1="{sy:.1f}" x2="{ax_x0:.1f}" y2="{sy:.1f}" '
                     f'stroke="#1f2937" stroke-width="{SW(0.8)}"/>')
            c.append(f'<text x="{ax_x0 - 6.0:.1f}" y="{sy:.1f}" font-size="{9.0 * PT:.2f}" '
                     f'fill="#1f2937" text-anchor="end" dominant-baseline="central">{esc(ylab)}</text>')
        if minor_yticks:
            for yv in minor_yticks:
                _, sy = d2svg(ax, 0, yv)
                c.append(f'<line x1="{ax_x0 - 2.0:.1f}" y1="{sy:.1f}" x2="{ax_x0:.1f}" y2="{sy:.1f}" '
                         f'stroke="#1f2937" stroke-width="{SW(0.6)}"/>')
        yl_bb = ax.yaxis.label.get_window_extent(renderer)
        yl_x = SX(0.5 * (yl_bb.x0 + yl_bb.x1))
        yl_y = 0.5 * (ax_y0 + ax_y1)
        c.append(f'<text x="{yl_x:.1f}" y="{yl_y:.1f}" font-size="{9.0 * PT:.2f}" fill="#1f2937" '
                 f'text-anchor="middle" dominant-baseline="central" '
                 f'transform="rotate(-90 {yl_x:.1f} {yl_y:.1f})">{esc(ylabel_str)}</text>')
        t_lines = [ln for ln in title_str.split("\n") if ln != ""]
        t_cx = 0.5 * (ax_x0 + ax_x1)
        t_fs = title_fs * PT
        t_lh = t_fs * 1.22
        # Anchor top of title block consistently above ax_y0
        t_top_y = ax_y0 - 24.5 if len(title_str.split("\n")) > 1 else ax_y0 - 11.5
        for k, ln in enumerate(t_lines):
            cur_fs = (sub_fs * PT) if (k > 0 and sub_fs is not None) else t_fs
            cur_col = sub_color if k > 0 else "#111827"
            c.append(f'<text x="{t_cx:.1f}" y="{t_top_y + k * t_lh:.1f}" font-size="{cur_fs:.2f}" '
                     f'font-weight="bold" fill="{cur_col}" text-anchor="middle">{esc(ln)}</text>')
        return "\n".join(c), ax_x0, ax_x1, ax_y0, ax_y1

    s = [f'<rect x="0" y="0" width="{vw}" height="{vh}" fill="#ffffff"/>']

    # Top clickable data-availability legend (synced with `.lg-row` filter toggles)
    leg_items = [
        ("open", OPEN_C[2], "open data"),
        ("partial", OPEN_C[1], "data on request"),
        ("closed", OPEN_C[0], "closed data"),
        ("unknown", OPEN_C["P"], "preprint"),
        ("reanalysis", OPEN_C["R"], "reanalysis on open data"),
    ]
    leg_y = 18.0
    leg_xs = [605.0, 740.0, 910.0, 1050.0, 1170.0]
    for (val, col, lab), lx in zip(leg_items, leg_xs):
        s.append(f'<g class="lg-row" data-dim="data_availability" data-val="{val}" style="cursor:pointer">')
        s.append(f'<rect x="{lx:.1f}" y="{leg_y - 5.5:.1f}" width="17" height="10" '
                 f'fill="{col}" fill-opacity="0.35" stroke="{col}" stroke-width="{SW(0.8)}"/>')
        s.append(f'<text x="{lx + 23:.1f}" y="{leg_y:.1f}" font-size="{8.4 * PT:.2f}" '
                 f'fill="#1f2937" dominant-baseline="central">{esc(lab)}</text>')
        s.append("</g>")
    # 6th legend item: our reproduction (matches Figure 2's lower-half bracket color)
    rep_c = "#3C5488"
    repro_url = "https://github.com/Physics-Lee/communication-neuroprostheses-reproduction"
    s.append(f'<a href="{repro_url}" target="_blank" rel="noopener" style="text-decoration:none">')
    s.append(f'<rect x="1395.0" y="{leg_y - 5.5:.1f}" width="17" height="10" '
             f'fill="{rep_c}" fill-opacity="0.35" stroke="{rep_c}" stroke-width="{SW(0.8)}"/>')
    s.append(f'<text x="1418.0" y="{leg_y:.1f}" font-size="{8.4 * PT:.2f}" '
             f'fill="#1f2937" dominant-baseline="central">our reproduction</text>')
    s.append("</a>")

    # ---- Panel (a) ----
    minor_locs = [m * 10 ** k for k in range(0, 6) for m in range(2, 10) if m * 10 ** k >= 2.5]
    ch_a, a_x0, a_x1, a_y0, a_y1 = draw_axes_chrome(
        ax1,
        [10 ** k for k in range(0, 7)],
        ["1", "10", "100", "1,000", "10,000", "100,000", "1,000,000"],
        "vocabulary size",
        f"(a) Maximum vocabulary size used in the {len(rows_a)} English communication-neuroprostheses studies",
        title_fs=10.0,
        minor_yticks=minor_locs,
    )
    s.append(ch_a)

    for anchor_v, anchor_lab, above in (
            (11100, "11,100 word families", False),
            (42000, "42,000 lemmas", False),
            (71400, "71,400 word forms", True)):
        _, sy_line = d2svg(ax1, 0, anchor_v)
        s.append(f'<line x1="{a_x0:.1f}" y1="{sy_line:.1f}" x2="{a_x1:.1f}" y2="{sy_line:.1f}" '
                 f'stroke="#7F7F7F" stroke-width="{SW(0.9)}" stroke-dasharray="5.5,2.8"/>')
        lx, ly = d2svg(ax1, -0.55, anchor_v * (1.12 if above else 0.72))
        s.append(multiline_svg(
            lx, ly, anchor_lab + " (Brysbaert et al., 2016)",
            7.0, "#555555", anchor="start", va="bottom" if above else "top"
        ))

    for x, r in enumerate(rows_a):
        v = r["vocab_num"]
        is_open = r["is_open_vocab"]
        col = OPEN_C[r["openness_code"]]
        bx0, by0 = d2svg(ax1, x - 0.38, v)
        bx1, by1 = d2svg(ax1, x + 0.38, 2.5)
        bw, bh = bx1 - bx0, by1 - by0
        cx = 0.5 * (bx0 + bx1)
        dash = ' stroke-dasharray="4.2,2.8"' if is_open else ""
        fop = "0.18" if is_open else "0.35"
        lbl_clean = r["label"].replace("\n", " ")
        s.append(f'<g class="f2-slot" data-id="{r["id"]}">')
        s.append(f'<title>{esc(lbl_clean)} \u2014 vocabulary: {esc(r["vocab_display"])} (click to inspect study)</title>')
        s.append(f'<rect class="f2-hit" x="{bx0 - 3:.1f}" y="{a_y0:.1f}" width="{bw + 6:.1f}" height="{a_y1 - a_y0 + 135:.1f}"/>')
        s.append(f'<rect class="f2-bar" x="{bx0:.1f}" y="{by0:.1f}" width="{bw:.1f}" height="{bh:.1f}" '
                 f'fill="{col}" fill-opacity="{fop}" stroke="{col}" stroke-width="{SW(0.8)}"{dash}/>')
        _, vy = d2svg(ax1, x, v * 1.15)
        s.append(multiline_svg(cx, vy, r["vocab_display"], 7.6, "#1f2937", anchor="middle", va="bottom", cls="f2-val"))
        how = open_how.get(r["study_key"]) if is_open else None
        if how is not None:
            _, hy = d2svg(ax1, x, v * 0.85)
            s.append(multiline_svg(cx, hy, how, 4.6, "#555555", anchor="middle", va="top"))
        s.append(f'<line x1="{cx:.1f}" y1="{a_y1:.1f}" x2="{cx:.1f}" y2="{a_y1 + 3.5:.1f}" '
                 f'stroke="#1f2937" stroke-width="{SW(0.8)}"/>')
        s.append(rot_xtick_svg(cx, a_y1 + 6.0, r["label"], 6.7))
        s.append("</g>")

    # ---- Green Funnel (Panel a -> Panels b-d) ----
    _, sy_bus = f2svg(0, y_bus)
    _, sy_apex = f2svg(0, y_apex)
    _, sy_arm = f2svg(0, y_arm)
    _, sy_tip = f2svg(0, y_tip)
    sx_b0, _ = f2svg(brace_x0, 0)
    sx_b1, _ = f2svg(brace_x1, 0)
    sx_bm, _ = f2svg(brace_xm, 0)

    for sid, x_pt, y_top in drop_points:
        sx_pt, sy_top = f2svg(x_pt, y_top)
        s.append(f'<g class="f2-drop" data-id="{sid}">')
        s.append(f'<line x1="{sx_pt:.1f}" y1="{sy_top:.1f}" x2="{sx_pt:.1f}" y2="{sy_bus:.1f}" '
                 f'stroke="{guide_c}" stroke-width="{SW(1.15)}" stroke-linecap="round" opacity="0.85"/>')
        s.append(f'<circle cx="{sx_pt:.1f}" cy="{sy_top:.1f}" r="2.4" fill="{guide_c}" opacity="0.9"/>')
        s.append("</g>")

    sx_first, _ = f2svg(drop_points[0][1], 0)
    sx_last, _ = f2svg(drop_points[-1][1], 0)
    s.append(f'<line x1="{sx_first:.1f}" y1="{sy_bus:.1f}" x2="{sx_last:.1f}" y2="{sy_bus:.1f}" '
             f'stroke="{guide_c}" stroke-width="{SW(1.25)}" stroke-linecap="round" opacity="0.88"/>')
    s.append(f'<line x1="{sx_bm:.1f}" y1="{sy_bus:.1f}" x2="{sx_bm:.1f}" y2="{sy_apex:.1f}" '
             f'stroke="{guide_c}" stroke-width="{SW(1.4)}" stroke-linecap="round" opacity="0.92"/>')

    dx_c = 0.016 * vw
    brace_d = (
        f"M {sx_b0:.1f} {sy_tip:.1f} "
        f"C {sx_b0:.1f} {sy_arm:.1f}, {sx_b0 + 0.35 * dx_c:.1f} {sy_arm:.1f}, {sx_b0 + dx_c:.1f} {sy_arm:.1f} "
        f"L {sx_bm - dx_c:.1f} {sy_arm:.1f} "
        f"C {sx_bm - 0.35 * dx_c:.1f} {sy_arm:.1f}, {sx_bm:.1f} {sy_arm:.1f}, {sx_bm:.1f} {sy_apex:.1f} "
        f"C {sx_bm:.1f} {sy_arm:.1f}, {sx_bm + 0.35 * dx_c:.1f} {sy_arm:.1f}, {sx_bm + dx_c:.1f} {sy_arm:.1f} "
        f"L {sx_b1 - dx_c:.1f} {sy_arm:.1f} "
        f"C {sx_b1 - 0.35 * dx_c:.1f} {sy_arm:.1f}, {sx_b1:.1f} {sy_arm:.1f}, {sx_b1:.1f} {sy_tip:.1f}"
    )
    s.append(f'<path d="{brace_d}" fill="none" stroke="{guide_c}" stroke-width="{SW(1.45)}" '
             f'stroke-linecap="round" stroke-linejoin="round" opacity="0.92"/>')
    sx_ftxt, sy_ftxt = f2svg(brace_xm - 0.008, 0.5 * (y_bus + y_arm) + 0.001)
    s.append(multiline_svg(
        sx_ftxt, sy_ftxt,
        "Filtered to 8 open-data studies exceeding native-speaker vocabulary (>71,400 word forms)",
        7.4, guide_c, anchor="end", va="center", weight="bold"
    ))

    # ---- Panel (b) ----
    ch_b, b_x0, b_x1, b_y0, b_y1 = draw_axes_chrome(
        ax_b,
        list(range(0, 12001, 2000)),
        [f"{v:,}" for v in range(0, 12001, 2000)],
        "released sentences",
        "(b) Open-sourced dataset size\n",
        title_fs=9.8,
    )
    s.append(ch_b)
    for x, r in enumerate(rows_b):
        n_sent = r["released_sentences"]
        col = OPEN_C[r["openness_code"]]
        is_star = r["id"] in STAR_IDS
        lbl_txt = ("\u2605 " + r["label"]) if is_star else r["label"]
        bx0, by0 = d2svg(ax_b, x - 0.38, n_sent)
        bx1, by1 = d2svg(ax_b, x + 0.38, 0)
        bw, bh = bx1 - bx0, by1 - by0
        cx = 0.5 * (bx0 + bx1)
        s.append(f'<g class="f2-slot" data-id="{r["id"]}">')
        s.append(f'<title>{esc(r["label"])} \u2014 {esc(r["sentences_display"])} released sentences (click to inspect study)</title>')
        s.append(f'<rect class="f2-hit" x="{bx0 - 3:.1f}" y="{b_y0:.1f}" width="{bw + 6:.1f}" height="{b_y1 - b_y0 + 125:.1f}"/>')
        s.append(f'<rect class="f2-bar" x="{bx0:.1f}" y="{by0:.1f}" width="{bw:.1f}" height="{bh:.1f}" '
                 f'fill="{col}" fill-opacity="0.35" stroke="{col}" stroke-width="{SW(0.8)}"/>')
        _, vy = d2svg(ax_b, x, n_sent + 220)
        s.append(multiline_svg(cx, vy, r["sentences_display"], 7.3, "#1f2937",
                               anchor="middle", va="bottom", cls="f2-val"))
        _, ny = d2svg(ax_b, x, n_sent + 1150)
        s.append(multiline_svg(cx, ny, r["span_note"], 4.6, "#555555", anchor="middle", va="bottom"))
        s.append(f'<line x1="{cx:.1f}" y1="{b_y1:.1f}" x2="{cx:.1f}" y2="{b_y1 + 3.5:.1f}" '
                 f'stroke="#1f2937" stroke-width="{SW(0.8)}"/>')
        s.append(rot_xtick_svg(cx, b_y1 + 6.0, lbl_txt, 6.4))
        s.append("</g>")

    # ---- Panel (c) ----
    ch_c, c_x0, c_x1, c_y0, c_y1 = draw_axes_chrome(
        ax2,
        list(range(0, 101, 25)),
        [str(v) for v in range(0, 101, 25)],
        "word error rate (%)",
        "(c) Word error rate (online)\n"
        "\u2605 3 studies satisfy: Open Data, Vocab > 71.4k, WER < 12.8%",
        title_fs=9.8,
    )
    s.append(ch_c)
    for asr_v in (2.5, 5.8, 12.8):
        _, sy_l = d2svg(ax2, 0, asr_v)
        s.append(f'<line x1="{c_x0:.1f}" y1="{sy_l:.1f}" x2="{c_x1:.1f}" y2="{sy_l:.1f}" '
                 f'stroke="#7F7F7F" stroke-width="{SW(0.9)}" stroke-dasharray="5.5,2.8"/>')
    for asr_v, asr_lab, ly, lx_lead, t_obj in c_badges:
        sx_ld, sy_l0 = d2svg(ax2, lx_lead, asr_v)
        _, sy_l1 = d2svg(ax2, lx_lead, ly - 0.2)
        s.append(f'<line x1="{sx_ld:.1f}" y1="{sy_l0:.1f}" x2="{sx_ld:.1f}" y2="{sy_l1:.1f}" '
                 f'stroke="#999999" stroke-width="{SW(0.55)}"/>')
        bp = t_obj.get_bbox_patch().get_window_extent(renderer)
        s.append(f'<rect x="{SX(bp.x0):.1f}" y="{SY(bp.y1):.1f}" width="{bp.width * 0.5:.1f}" '
                 f'height="{bp.height * 0.5:.1f}" rx="2.5" fill="#ffffff" fill-opacity="0.88"/>')
        tx, ty = d2svg(ax2, 5.50, ly)
        s.append(multiline_svg(tx, ty, asr_lab, 5.6, "#666666", anchor="end", va="bottom"))
    s.append(multiline_svg(
        c_x0 + 0.98 * (c_x1 - c_x0), c_y0 + 0.015 * (c_y1 - c_y0),
        dagger_note, 6.0, "#555555", anchor="end", va="top"
    ))
    for x, r in enumerate(rows_cd):
        wmean = r["wer_mean"]
        wsd = r["wer_sd"]
        col = OPEN_C[r["openness_code"]]
        is_star = r["id"] in STAR_IDS
        lbl_txt = ("\u2605 " + r["label"]) if is_star else r["label"]
        bx0, by0 = d2svg(ax2, x - 0.38, wmean)
        bx1, by1 = d2svg(ax2, x + 0.38, 0)
        bw, bh = bx1 - bx0, by1 - by0
        cx = 0.5 * (bx0 + bx1)
        top = wmean + wsd if wsd is not None else wmean
        note = f"{wmean:.1f}\u00b1{wsd:.1f}" if wsd is not None else f"{wmean:.1f}"
        s.append(f'<g class="f2-slot" data-id="{r["id"]}">')
        s.append(f'<title>{esc(r["label"])} \u2014 online WER: {esc(note)}% (click to inspect study)</title>')
        s.append(f'<rect class="f2-hit" x="{bx0 - 3:.1f}" y="{c_y0:.1f}" width="{bw + 6:.1f}" height="{c_y1 - c_y0 + 125:.1f}"/>')
        s.append(f'<rect class="f2-bar" x="{bx0:.1f}" y="{by0:.1f}" width="{bw:.1f}" height="{bh:.1f}" '
                 f'fill="{col}" fill-opacity="0.35" stroke="{col}" stroke-width="{SW(0.8)}"/>')
        if wsd is not None:
            _, ey_hi = d2svg(ax2, x, wmean + wsd)
            _, ey_lo = d2svg(ax2, x, wmean - wsd)
            s.append(f'<line x1="{cx:.1f}" y1="{ey_hi:.1f}" x2="{cx:.1f}" y2="{ey_lo:.1f}" '
                     f'stroke="#333333" stroke-width="{SW(1.0)}"/>')
            s.append(f'<line x1="{cx - 2.8:.1f}" y1="{ey_hi:.1f}" x2="{cx + 2.8:.1f}" y2="{ey_hi:.1f}" '
                     f'stroke="#333333" stroke-width="{SW(1.0)}"/>')
            s.append(f'<line x1="{cx - 2.8:.1f}" y1="{ey_lo:.1f}" x2="{cx + 2.8:.1f}" y2="{ey_lo:.1f}" '
                     f'stroke="#333333" stroke-width="{SW(1.0)}"/>')
        _, vy = d2svg(ax2, x, top + 1.6)
        s.append(multiline_svg(cx, vy, note, 6.6, "#1f2937",
                               anchor="middle", va="bottom", cls="f2-val"))
        _, ny = d2svg(ax2, x, top + 7.0)
        s.append(multiline_svg(cx, ny, f"vocab {r['vocab_tag']}\n{r['wer_caliber']}", 4.6, "#555555", anchor="middle", va="bottom"))
        if is_star:
            _, sy_st = d2svg(ax2, x, top + 18.8)
            s.append(multiline_svg(cx, sy_st, "\u2605", 7.8, "#1f2937", anchor="middle", va="bottom"))
        s.append(f'<line x1="{cx:.1f}" y1="{c_y1:.1f}" x2="{cx:.1f}" y2="{c_y1 + 3.5:.1f}" '
                 f'stroke="#1f2937" stroke-width="{SW(0.8)}"/>')
        s.append(rot_xtick_svg(cx, c_y1 + 6.0, lbl_txt, 6.4))
        s.append("</g>")

    # ---- Panel (d) ----
    ch_d, d_x0, d_x1, d_y0, d_y1 = draw_axes_chrome(
        ax3,
        list(range(0, 201, 50)),
        [str(v) for v in range(0, 201, 50)],
        "words per minute",
        "(d) Communication speed (online)\nn/r: speed not reported",
        title_fs=9.8,
    )
    s.append(ch_d)
    for spd_v in (160, 51.6, 36.2, 10, 8):
        _, sy_l = d2svg(ax3, 0, spd_v)
        s.append(f'<line x1="{d_x0:.1f}" y1="{sy_l:.1f}" x2="{d_x1:.1f}" y2="{sy_l:.1f}" '
                 f'stroke="#7F7F7F" stroke-width="{SW(0.9)}" stroke-dasharray="5.5,2.8"/>')
    sx_ld8, sy_8_0 = d2svg(ax3, 5.5, 8)
    _, sy_8_1 = d2svg(ax3, 5.5, 19.8)
    s.append(f'<line x1="{sx_ld8:.1f}" y1="{sy_8_0:.1f}" x2="{sx_ld8:.1f}" y2="{sy_8_1:.1f}" '
             f'stroke="#999999" stroke-width="{SW(0.55)}"/>')
    for ly_d, spd_lab, t_obj in d_badges:
        bp = t_obj.get_bbox_patch().get_window_extent(renderer)
        s.append(f'<rect x="{SX(bp.x0):.1f}" y="{SY(bp.y1):.1f}" width="{bp.width * 0.5:.1f}" '
                 f'height="{bp.height * 0.5:.1f}" rx="2.5" fill="#ffffff" fill-opacity="0.88"/>')
        tx, ty = d2svg(ax3, LX, ly_d)
        s.append(multiline_svg(tx, ty, spd_lab, 5.6, "#666666", anchor="start", va="bottom"))
    s.append(multiline_svg(
        d_x0 + 0.98 * (d_x1 - d_x0), d_y0 + 0.015 * (d_y1 - d_y0),
        dagger_note, 6.0, "#555555", anchor="end", va="top"
    ))
    for x, r in enumerate(rows_cd):
        spd = r["speed_wpm_mean"]
        spdsd = r["speed_wpm_sd"]
        col = OPEN_C[r["openness_code"]]
        is_star = r["id"] in STAR_IDS
        lbl_txt = ("\u2605 " + r["label"]) if is_star else r["label"]
        bx0, _ = d2svg(ax3, x - 0.38, 0)
        bx1, by1 = d2svg(ax3, x + 0.38, 0)
        bw = bx1 - bx0
        cx = 0.5 * (bx0 + bx1)
        s.append(f'<g class="f2-slot" data-id="{r["id"]}">')
        if spd is None:
            s.append(f'<title>{esc(r["label"])} \u2014 speed not reported (click to inspect study)</title>')
            s.append(f'<rect class="f2-hit" x="{bx0 - 3:.1f}" y="{d_y0:.1f}" width="{bw + 6:.1f}" height="{d_y1 - d_y0 + 125:.1f}"/>')
            _, nry = d2svg(ax3, x, 1.2)
            s.append(multiline_svg(cx, nry, "n/r", 5.6, "#777777", anchor="middle", va="bottom", cls="f2-val"))
        else:
            _, by0 = d2svg(ax3, x - 0.38, spd)
            bh = by1 - by0
            top = spd + spdsd if spdsd is not None else spd
            note = f"{spd:.1f}\u00b1{spdsd:.1f}" if spdsd is not None else f"{spd:.1f}"
            s.append(f'<title>{esc(r["label"])} \u2014 online speed: {esc(note)} wpm (click to inspect study)</title>')
            s.append(f'<rect class="f2-hit" x="{bx0 - 3:.1f}" y="{d_y0:.1f}" width="{bw + 6:.1f}" height="{d_y1 - d_y0 + 125:.1f}"/>')
            s.append(f'<rect class="f2-bar" x="{bx0:.1f}" y="{by0:.1f}" width="{bw:.1f}" height="{bh:.1f}" '
                     f'fill="{col}" fill-opacity="0.35" stroke="{col}" stroke-width="{SW(0.8)}"/>')
            if spdsd is not None:
                _, ey_hi = d2svg(ax3, x, spd + spdsd)
                _, ey_lo = d2svg(ax3, x, spd - spdsd)
                s.append(f'<line x1="{cx:.1f}" y1="{ey_hi:.1f}" x2="{cx:.1f}" y2="{ey_lo:.1f}" '
                         f'stroke="#333333" stroke-width="{SW(1.0)}"/>')
                s.append(f'<line x1="{cx - 2.8:.1f}" y1="{ey_hi:.1f}" x2="{cx + 2.8:.1f}" y2="{ey_hi:.1f}" '
                         f'stroke="#333333" stroke-width="{SW(1.0)}"/>')
                s.append(f'<line x1="{cx - 2.8:.1f}" y1="{ey_lo:.1f}" x2="{cx + 2.8:.1f}" y2="{ey_lo:.1f}" '
                         f'stroke="#333333" stroke-width="{SW(1.0)}"/>')
            _, vy = d2svg(ax3, x, top * 1.06 + 2.0)
            s.append(multiline_svg(cx, vy, note, 6.8, "#1f2937",
                                   anchor="middle", va="bottom", cls="f2-val"))
            _, ny = d2svg(ax3, x, max(top * 1.31, top * 1.06 + 8.0) + 2.0)
            s.append(multiline_svg(cx, ny, f"vocab {r['vocab_tag']}\n{r['speed_caliber']}", 4.6, "#555555", anchor="middle", va="bottom"))
        s.append(f'<line x1="{cx:.1f}" y1="{d_y1:.1f}" x2="{cx:.1f}" y2="{d_y1 + 3.5:.1f}" '
                 f'stroke="#1f2937" stroke-width="{SW(0.8)}"/>')
        s.append(rot_xtick_svg(cx, d_y1 + 6.0, lbl_txt, 6.4))
        s.append("</g>")

    # ---- Second Curly Bracket (gathering Panel c bars -> reproduction repository) ----
    lbl_y0_rep = min(
        inv_fig.transform_bbox(lbl.get_window_extent(renderer)).y0
        for lbl in (*ax_b.get_xticklabels(), *ax2.get_xticklabels(), *ax3.get_xticklabels())
    )
    _, sy_lbl_bot = f2svg(0, lbl_y0_rep)
    drop_pts_c = []
    for i, lbl in enumerate(ax2.get_xticklabels()):
        bb_c = inv_fig.transform_bbox(lbl.get_window_extent(renderer))
        x_anchor_c = 0.5 * (bb_c.x0 + bb_c.x1)
        bar_x_c = inv_fig.transform(ax2.transData.transform((i, 0)))[0]
        x_pt_c = 0.65 * bar_x_c + 0.35 * x_anchor_c
        sx_pt_c, sy_top_c = f2svg(x_pt_c, bb_c.y0 - 0.0042)
        drop_pts_c.append((rows_cd[i]["id"], sx_pt_c, sy_top_c))

    cor_rep_svg = 84.0
    sy_bus_c = sy_lbl_bot + 0.28 * cor_rep_svg
    sy_apex_r = sy_lbl_bot + 0.42 * cor_rep_svg
    sy_arm_r = sy_lbl_bot + 0.70 * cor_rep_svg
    sy_tip_r = sy_lbl_bot + 0.95 * cor_rep_svg

    for sid_c, sx_pt_c, sy_top_c in drop_pts_c:
        is_star = sid_c in STAR_IDS
        s.append(f'<g class="f2-drop" data-id="{sid_c}">')
        if is_star:
            s.append(f'<line x1="{sx_pt_c:.1f}" y1="{sy_top_c:.1f}" x2="{sx_pt_c:.1f}" y2="{sy_bus_c:.1f}" '
                     f'stroke="{rep_c}" stroke-width="{SW(1.85)}" stroke-linecap="round" opacity="0.98"/>')
            s.append(f'<circle cx="{sx_pt_c:.1f}" cy="{sy_top_c:.1f}" r="3.2" fill="{rep_c}" opacity="0.98"/>')
            s.append(f'<circle cx="{sx_pt_c:.1f}" cy="{sy_top_c:.1f}" r="1.2" fill="#ffffff"/>')
        else:
            s.append(f'<line x1="{sx_pt_c:.1f}" y1="{sy_top_c:.1f}" x2="{sx_pt_c:.1f}" y2="{sy_bus_c:.1f}" '
                     f'stroke="{rep_c}" stroke-width="{SW(0.95)}" stroke-dasharray="3.2,2.4" stroke-linecap="round" opacity="0.45"/>')
            s.append(f'<circle cx="{sx_pt_c:.1f}" cy="{sy_top_c:.1f}" r="1.8" fill="{rep_c}" opacity="0.48"/>')
        s.append("</g>")

    s.append(f'<line x1="{drop_pts_c[0][1]:.1f}" y1="{sy_bus_c:.1f}" '
             f'x2="{drop_pts_c[-1][1]:.1f}" y2="{sy_bus_c:.1f}" '
             f'stroke="{rep_c}" stroke-width="{SW(1.25)}" stroke-linecap="round" opacity="0.88"/>')
    s.append(f'<line x1="{sx_bm:.1f}" y1="{sy_bus_c:.1f}" x2="{sx_bm:.1f}" y2="{sy_apex_r:.1f}" '
             f'stroke="{rep_c}" stroke-width="{SW(1.4)}" stroke-linecap="round" opacity="0.92"/>')

    brace_r_d = (
        f"M {sx_b0:.1f} {sy_tip_r:.1f} "
        f"C {sx_b0:.1f} {sy_arm_r:.1f}, {sx_b0 + 0.35 * dx_c:.1f} {sy_arm_r:.1f}, {sx_b0 + dx_c:.1f} {sy_arm_r:.1f} "
        f"L {sx_bm - dx_c:.1f} {sy_arm_r:.1f} "
        f"C {sx_bm - 0.35 * dx_c:.1f} {sy_arm_r:.1f}, {sx_bm:.1f} {sy_arm_r:.1f}, {sx_bm:.1f} {sy_apex_r:.1f} "
        f"C {sx_bm:.1f} {sy_arm_r:.1f}, {sx_bm + 0.35 * dx_c:.1f} {sy_arm_r:.1f}, {sx_bm + dx_c:.1f} {sy_arm_r:.1f} "
        f"L {sx_b1 - dx_c:.1f} {sy_arm_r:.1f} "
        f"C {sx_b1 - 0.35 * dx_c:.1f} {sy_arm_r:.1f}, {sx_b1:.1f} {sy_arm_r:.1f}, {sx_b1:.1f} {sy_tip_r:.1f}"
    )
    s.append(f'<path d="{brace_r_d}" fill="none" stroke="{rep_c}" stroke-width="{SW(1.45)}" '
             f'stroke-linecap="round" stroke-linejoin="round" opacity="0.92"/>')
    s.append(multiline_svg(
        sx_bm - 0.012 * vw, 0.5 * (sy_bus_c + sy_arm_r),
        "Our reproduction on a 64 GB RAM / RTX 5060 Ti (16 GB) home PC",
        7.4, rep_c, anchor="end", va="center", weight="bold"
    ))

    # Reproduction callout under the second bracket
    box_w, box_h = 880.0, 42.0
    box_x = sx_bm - 0.5 * box_w
    box_y = sy_tip_r + 4.0
    s.append(f'<a href="{repro_url}" target="_blank" rel="noopener" class="f2-repro-link" style="text-decoration:none">')
    s.append(f'<rect x="{box_x:.1f}" y="{box_y:.1f}" width="{box_w:.1f}" height="{box_h:.1f}" rx="9" '
             f'fill="#f5f7fc" stroke="{rep_c}" stroke-opacity="0.38" stroke-width="1.1"/>')
    s.append(f'<text x="{sx_bm:.1f}" y="{box_y + 16.0:.1f}" font-size="{8.2 * PT:.2f}" '
             f'font-weight="bold" fill="{rep_c}" text-anchor="middle" dominant-baseline="central">'
             f'Please refer to our reproduction repositories (<tspan text-decoration="underline">link \u2197</tspan>).</text>')
    s.append(f'<text x="{sx_bm:.1f}" y="{box_y + 31.5:.1f}" font-size="{7.5 * PT:.2f}" '
             f'fill="#4b5563" text-anchor="middle" dominant-baseline="central">'
             f'We promised to update the timeline and reproduction for future communication neuroprostheses studies.</text>')
    s.append("</a>")

    plt.close(fig)
    return "\n".join(s), vw, vh


def _poly_pts(cx: float, cy: float, r: float, n: int, rot_deg: float = 90.0) -> str:
    """Build an SVG polygon ``points`` attribute for a regular n-gon.

    Parameters
    ----------
    cx, cy : float
        Center coordinates of the polygon in SVG user units.
    r : float
        Circumscribed radius of the polygon.
    n : int
        Number of polygon vertices (3 = triangle, 5 = pentagon, 6 = hexagon).
    rot_deg : float, default=90.0
        Rotation of the first vertex counter-clockwise from the +x axis, in degrees.

    Returns
    -------
    str
        Space-separated ``x,y`` vertex list suitable for a ``<polygon points>``
        attribute.
    """
    pts = []
    for i in range(n):
        a = math.radians(rot_deg + i * 360.0 / n)
        pts.append(f"{cx + r * math.cos(a):.2f},{cy - r * math.sin(a):.2f}")
    return " ".join(pts)


def _build_chrome_svg(geom: dict[str, Any]) -> str:
    """Render the static SVG chrome (axis, tracks, legends, notes) for the HTML edition.

    Parameters
    ----------
    geom : dict of str to Any
        Geometry payload captured from the Matplotlib pass, including the data
        limits (``x0``, ``x1``, ``y0``, ``y1``), SVG extents (``sx0``, ``sx1``,
        ``sy0``, ``sy1``), lane track rows, and the six equalized legend boxes.

    Returns
    -------
    str
        Concatenated SVG markup for every non-study element of the interactive
        figure, expressed in 0.5-scale pixel coordinates.
    """
    oc = {"open": "#00A087", "partial": "#c07f00", "closed": "#cc3324",
          "unknown": "#7F7F7F", "reanalysis": "#7B3294"}
    kx = (geom["sx1"] - geom["sx0"]) / (geom["x1"] - geom["x0"])
    ky = (geom["sy0"] - geom["sy1"]) / (geom["y1"] - geom["y0"])

    def X(v):
        return geom["sx0"] + (v - geom["x0"]) * kx

    def Y(v):
        # sy1 is the SVG y of data-y0 (bottom); ky is negative so data grows upward
        return geom["sy1"] + (v - geom["y0"]) * ky

    PT = 1.3889  # svg units per pt at 200 dpi, 0.5 scale
    SW = lambda pt: round(pt * PT, 2)
    s = []

    s.append(f'<rect x="0" y="0" width="{geom["sx1"] + 8:.0f}" height="{geom["sy1"] + 8:.0f}" fill="#ffffff"/>')

    # Year axis
    s.append(f'<line x1="{X(geom["x0"]):.1f}" y1="{Y(geom["yax"]):.1f}" '
             f'x2="{X(geom["vis_end"]):.1f}" y2="{Y(geom["yax"]):.1f}" stroke="#9aa3af" stroke-width="{SW(1.2)}"/>')
    for yr in range(2021, geom["max_yr_tick"]):
        s.append(f'<line x1="{X(yr):.1f}" y1="{Y(geom["yax"]):.1f}" '
                 f'x2="{X(yr):.1f}" y2="{Y(geom["yax"] - 0.15):.1f}" stroke="#9aa3af" stroke-width="{SW(1.2)}"/>')
        s.append(f'<text x="{X(yr):.1f}" y="{Y(-3.57):.1f}" font-size="{10 * PT:.1f}" fill="#6b7280" '
                 f'text-anchor="middle" dominant-baseline="central">{yr}</text>')

    # Lane tracks
    lane_label = {"speech": "Speech", "typing": "Typing", "handwriting": "Handwriting"}
    for lane, ty in geom["tracks"].items():
        s.append(f'<line x1="{X(geom["x0"]):.1f}" y1="{Y(ty):.1f}" x2="{X(geom["x1"] + 0.6):.1f}" '
                 f'y2="{Y(ty):.1f}" stroke="#e2e6ec" stroke-width="{SW(1.6)}"/>')
        s.append(f'<text x="{X(geom["x0"] - 0.18):.1f}" y="{Y(ty):.1f}" font-size="{11.5 * PT:.1f}" '
                 f'font-weight="bold" fill="#4b5563" text-anchor="end" dominant-baseline="central">'
                 f'{lane_label[lane]}</text>')

    # Legend boxes (clickable rows mirror the 6 figure legends; the decoded-output
    # and task-instruction boxes intentionally expose exactly two toggle options)
    def marker_svg(kind, color, cx, cy):
        if kind == "solid-circle":
            return (f'<circle cx="{cx}" cy="{cy}" r="5" fill="{color}" stroke="white" stroke-width="1"/>')
        if kind == "hollow-circle":
            return (f'<circle cx="{cx}" cy="{cy}" r="3.6" fill="white" stroke="{color}" stroke-width="1.8"/>')
        if kind == "solid-square":
            return (f'<rect x="{cx - 4}" y="{cy - 4}" width="8" height="8" fill="{color}" '
                    f'stroke="white" stroke-width="1"/>')
        if kind == "solid-hex":
            return f'<polygon points="{_poly_pts(cx, cy, 5, 6, 90)}" fill="{color}" stroke="white" stroke-width="1"/>'
        if kind == "solid-tri-v":
            return f'<polygon points="{_poly_pts(cx, cy, 5, 3, -90)}" fill="{color}" stroke="white" stroke-width="1"/>'
        if kind == "solid-tri-up":
            return f'<polygon points="{_poly_pts(cx, cy, 5.2, 3, 90)}" fill="{color}" stroke="white" stroke-width="1"/>'
        if kind == "hollow-tri-up":
            return (f'<polygon points="{_poly_pts(cx, cy, 4.4, 3, 90)}" fill="white" '
                    f'stroke="{color}" stroke-width="1"/>')
        if kind == "solid-diamond":
            return f'<polygon points="{_poly_pts(cx, cy, 4.6, 4, 90)}" fill="{color}" stroke="{color}" stroke-width="0.8"/>'
        if kind == "hollow-diamond":
            return (f'<polygon points="{_poly_pts(cx, cy, 4.6, 4, 90)}" fill="white" '
                    f'stroke="{color}" stroke-width="1"/>')
        if kind == "solid-pentagon":
            return f'<polygon points="{_poly_pts(cx, cy, 5, 5, 90)}" fill="{color}" stroke="{color}" stroke-width="0.8"/>'
        if kind == "hollow-pentagon":
            return (f'<polygon points="{_poly_pts(cx, cy, 5, 5, 90)}" fill="white" '
                    f'stroke="{color}" stroke-width="1"/>')
        raise ValueError(kind)

    GRAY = "#7F7F7F"
    LEG_DEFS = [
        ("data_availability", "data availability", [
            ("solid-circle", oc["open"], "open data", "open"),
            ("solid-circle", oc["partial"], "data on request", "partial"),
            ("solid-circle", oc["closed"], "closed data", "closed"),
            ("solid-circle", oc["unknown"], "preprint", "unknown"),
            ("solid-circle", oc["reanalysis"], "reanalysis on open data", "reanalysis"),
        ]),
        ("decoding_mode", "decoding mode", [
            ("solid-circle", GRAY, "online", "online"),
            ("hollow-circle", GRAY, "offline", "offline"),
        ]),
        ("electrode", "electrode type", [
            ("solid-circle", GRAY, "ECoG", "ecog"),
            ("solid-square", GRAY, "Utah array", "utah"),
            ("solid-hex", GRAY, "thin-film threads", "thin_film"),
            ("solid-tri-v", GRAY, "microwire array", "microwire"),
        ]),
        ("speech_ability", "speech ability", [
            ("solid-tri-up", "#1a1a1a", "participants who can speak naturally", "intact"),
            ("hollow-tri-up", GRAY, "participants who have vocal paralysis", "paralysis"),
        ]),
        ("task_instruction", "task instruction", [
            ("solid-diamond", "#1a1a1a", "inner speech", "inner"),
            ("hollow-diamond", "#1a1a1a", "overt/mouthing/intended", "intended"),
        ]),
        ("decoded_output", "decoded output", [
            ("solid-pentagon", "#1a1a1a", "brain-to-voice", "voice"),
            ("hollow-pentagon", "#1a1a1a", "brain-to-text", "text"),
        ]),
    ]

    for (dim, title, rows), box in zip(LEG_DEFS, geom["legend_boxes"]):
        bx, by, bw, bh = box["x"], box["y"], box["w"], box["h"]
        g = [f'<g class="legend" data-dim="{dim}">',
             f'<rect class="frame" x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="10" '
             f'fill="white" fill-opacity="0.97" stroke="#e2e6ec"/>',
             f'<text class="lg-title" x="{bx + 12:.1f}" y="{by + 15:.1f}" font-size="{7.4 * PT:.1f}" '
             f'font-weight="bold" letter-spacing="0.8" fill="#8a93a0">{title.upper()}</text>']
        ry = by + 32
        for kind, color, label, val in rows:
            g.append(f'<g class="lg-row" data-dim="{dim}" data-val="{val}">')
            g.append(marker_svg(kind, color, bx + 17, ry))
            g.append(f'<text x="{bx + 30:.1f}" y="{ry:.1f}" font-size="{7.8 * PT:.1f}" fill="#374151" '
                     f'dominant-baseline="central">{label}</text>')
            g.append('</g>')
            ry += 15.5
        g.append('</g>')
        s.extend(g)

    # Footer note + per-lane totals (JS keeps them in sync with the filter)
    s.append(f'<text id="total-note" x="{X(geom["x1"]):.1f}" y="{Y(-4.21):.1f}" font-size="{8.2 * PT:.1f}" '
             f'font-style="italic" fill="#3C5488" text-anchor="end" dominant-baseline="central"></text>')
    for lane, ty in geom["tracks"].items():
        s.append(f'<text id="lt-{lane}" x="{X(geom["x1"] - 0.03):.1f}" y="{Y(ty - 0.18):.1f}" '
                 f'font-size="{10 * PT:.1f}" font-weight="bold" fill="#3C5488" text-anchor="end"></text>')

    # Empty-state message (topmost)
    mid_x = 0.5 * (geom["sx0"] + geom["sx1"])
    mid_y = 0.5 * (geom["sy0"] + geom["sy1"])
    s.append(f'<g id="empty-msg" style="display:none">'
             f'<text x="{mid_x:.0f}" y="{mid_y:.0f}" font-size="16" fill="#8a93a0" '
             f'text-anchor="middle">No studies match the current filters</text></g>')
    return "\n".join(s)


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ · Interactive Edition</title>
<style>
  :root {
    --ink: #1f2937; --muted: #6b7280; --faint: #9aa3af;
    --line: #e5e7eb; --canvas: #f5f6f8; --card: #ffffff;
    --accent: #0f4c81; --accent-soft: rgba(15, 76, 129, .08);
  }
  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; }
  body { margin: 0; background: var(--canvas); color: var(--ink);
         font: 14px/1.45 "Segoe UI", "Helvetica Neue", Helvetica, Arial, sans-serif;
         -webkit-font-smoothing: antialiased; }
  .wrap { max-width: 1600px; margin: 0 auto; padding: 0 28px 80px; }

  /* ---------- header ---------- */
  header { position: sticky; top: 0; z-index: 50; background: rgba(245, 246, 248, .92);
           backdrop-filter: blur(6px); border-bottom: 1px solid var(--line);
           margin: 0 -28px; padding: 12px 28px; }
  .head-row { display: flex; flex-wrap: wrap; align-items: baseline; gap: 8px 18px; max-width: 1600px; margin: 0 auto; }
  h1 { font-size: 17px; font-weight: 650; margin: 0; letter-spacing: .01em; }
  .head-tools { margin-left: auto; display: flex; gap: 10px; align-items: center; }
  .stat { background: var(--card); border: 1px solid var(--line); border-radius: 999px;
          padding: 3px 12px; font-size: 12px; color: var(--muted); white-space: nowrap; }
  .stat b { color: var(--ink); font-weight: 650; }
  #search { border: 1px solid var(--line); border-radius: 999px; padding: 5px 14px;
            font-size: 12.5px; width: 230px; background: var(--card); color: var(--ink); outline: none; }
  #search:focus { border-color: var(--accent); }
  #reset { border: 1px solid var(--line); background: var(--card); border-radius: 999px;
           padding: 5px 14px; font-size: 12.5px; color: var(--muted); cursor: pointer; }
  #reset:hover { color: var(--ink); border-color: var(--faint); }

  /* ---------- filter bar ---------- */
  #panel { background: var(--card); border: 1px solid var(--line); border-radius: 14px;
           margin: 18px 0; padding: 14px 18px 10px; }
  .groups { display: flex; flex-wrap: wrap; gap: 10px 30px; }
  .fgroup { display: flex; flex-direction: column; gap: 5px; }
  .fg-title { font-size: 10.5px; font-weight: 700; letter-spacing: .08em;
              text-transform: uppercase; color: var(--faint); }
  .chips { display: flex; flex-wrap: wrap; gap: 5px; }
  .chip { display: inline-flex; align-items: center; gap: 6px; border: 1px solid var(--line);
          background: #fafbfc; color: var(--muted); border-radius: 999px;
          padding: 2.5px 11px; font-size: 12px; cursor: pointer; user-select: none;
          transition: all .12s ease; }
  .chip:hover { border-color: var(--faint); color: var(--ink); }
  .chip.on { background: var(--accent-soft); border-color: var(--accent); color: var(--accent); font-weight: 600; }
  .chip .sw { width: 9px; height: 9px; border-radius: 50%; flex: none; }
  .chip .mg { font-size: 11px; flex: none; opacity: .85; }
  .date-range { display: inline-flex; align-items: center; gap: 6px; border: 1px solid var(--line);
                background: #fafbfc; border-radius: 999px; padding: 2px 11px; font-size: 12px;
                color: var(--muted); transition: all .12s ease; }
  .date-range:hover { border-color: var(--faint); }
  .date-range:focus-within { border-color: var(--accent); color: var(--ink); }
  .date-range.filtered { background: var(--accent-soft); border-color: var(--accent); color: var(--accent); }
  .date-range input[type="date"] { border: none; background: transparent; color: var(--ink);
                                   font: inherit; font-size: 12px; padding: 0; outline: none;
                                   cursor: pointer; }
  .date-range.filtered input[type="date"] { color: var(--accent); font-weight: 600; }
  .date-sep { color: var(--faint); user-select: none; }
  #date-clear { border: none; background: none; color: var(--muted); font-size: 14px;
                line-height: 1; padding: 0 2px; cursor: pointer; display: none; }
  #date-clear:hover { color: var(--ink); }
  .date-range.filtered #date-clear { display: inline-block; color: var(--accent); }

  /* ---------- view mode ---------- */
  .bottom-row { display: flex; flex-wrap: wrap; gap: 14px; align-items: center;
                border-top: 1px solid var(--line); margin-top: 12px; padding-top: 10px; }
  .mode-lbl { font-size: 10.5px; font-weight: 700; letter-spacing: .08em;
              text-transform: uppercase; color: var(--faint); }
  #modebar { display: inline-flex; border: 1px solid var(--line); border-radius: 999px; overflow: hidden; }
  #modebar button { border: none; background: transparent; padding: 4px 14px; font-size: 12px;
                    color: var(--muted); cursor: pointer; }
  #modebar button + button { border-left: 1px solid var(--line); }
  #modebar button.on { background: var(--accent); color: #fff; font-weight: 600; }
  #pinned-only-btn { border: 1px solid rgba(0, 107, 89, .38); background: rgba(0, 160, 135, .08);
                     color: #005f4f; border-radius: 999px; padding: 4px 14px; font-size: 12px;
                     font-weight: 600; cursor: pointer; transition: all .12s ease; }
  #pinned-only-btn:hover { border-color: #006b59; background: rgba(0, 160, 135, .15); }
  #pinned-only-btn.on { background: #006b59; border-color: #006b59; color: #fff; }

  /* ---------- figure ---------- */
  #figbox, #benchbox { position: relative; background: var(--card); border: 1px solid var(--line); border-radius: 14px;
                       padding: 10px 10px 4px; }
  #figbox:fullscreen, #figbox:-webkit-full-screen, #figbox.is-fullscreen,
  #benchbox:fullscreen, #benchbox:-webkit-full-screen, #benchbox.is-fullscreen {
    background: var(--card); border: none; border-radius: 0; padding: 14px 18px;
    display: flex; align-items: center; justify-content: center; overflow: auto;
    width: 100vw; height: 100vh;
  }
  #figbox.is-fullscreen, #benchbox.is-fullscreen { position: fixed; inset: 0; z-index: 9999; }
  #tl-svg, #f2-svg { width: 100%; height: auto; display: block; border-radius: 8px; }
  #figbox:fullscreen #tl-svg, #figbox:-webkit-full-screen #tl-svg, #figbox.is-fullscreen #tl-svg,
  #benchbox:fullscreen #f2-svg, #benchbox:-webkit-full-screen #f2-svg, #benchbox.is-fullscreen #f2-svg {
    width: 100%; height: 100%; max-width: 100%; max-height: 100%;
  }
  #fs-btn, #f2-fs-btn { position: absolute; top: 14px; right: 16px; z-index: 10;
                        display: inline-flex; align-items: center; gap: 6px;
                        border: 1px solid var(--line); background: rgba(255, 255, 255, .92);
                        backdrop-filter: blur(4px); border-radius: 999px; padding: 5px 12px;
                        font-size: 12px; font-weight: 500; color: var(--muted); cursor: pointer;
                        box-shadow: 0 1px 4px rgba(15, 36, 64, .06); transition: all .12s ease; }
  #fs-btn:hover, #f2-fs-btn:hover { color: var(--ink); border-color: var(--faint);
                                    box-shadow: 0 2px 8px rgba(15, 36, 64, .10); }
  #fs-btn svg, #f2-fs-btn svg { width: 13px; height: 13px; stroke: currentColor; stroke-width: 1.8;
                                fill: none; stroke-linecap: round; stroke-linejoin: round; flex: none; }
  #tl-svg text, #f2-svg text { font-family: "Segoe UI", "DejaVu Sans", Arial, sans-serif; }
  .f2-slot { cursor: pointer; transition: opacity .15s ease; }
  .f2-slot .f2-hit { fill: transparent; }
  .f2-slot .f2-bar { transition: stroke-width .12s ease, fill-opacity .12s ease; }
  .f2-slot:hover .f2-bar, .f2-slot.hovered .f2-bar, .f2-slot.selected .f2-bar {
    stroke-width: 2.3px; fill-opacity: 0.56;
  }
  .f2-slot:hover .f2-lbl, .f2-slot.hovered .f2-lbl, .f2-slot.selected .f2-lbl,
  .f2-slot:hover .f2-val, .f2-slot.hovered .f2-val, .f2-slot.selected .f2-val {
    fill: var(--accent); font-weight: 700;
  }
  .f2-drop { transition: opacity .15s ease; }
  .f2-repro-link rect { transition: stroke-opacity .12s ease, fill .12s ease; }
  .f2-repro-link:hover rect { stroke-opacity: .85; fill: #eef2fb; }
  .card { cursor: pointer; }
  .card text { paint-order: stroke; stroke: rgba(255,255,255,.94); stroke-width: 3px;
               transition: fill .12s ease; }
  .card:hover .t-name, .card.selected .t-name { fill: var(--accent); }
  .card.selected .t-name { font-weight: 700; }
  .mark { cursor: pointer; }
  .legend { cursor: pointer; }
  .legend .frame { transition: stroke .12s ease; }
  .legend:hover .frame { stroke: var(--faint); }
  .lg-row text { transition: fill .12s ease; }
  .lg-row:hover text { fill: var(--ink); }
  .lg-row.off text { fill: #c3c9d1; text-decoration: line-through; text-decoration-color: #c3c9d1; }
  .lg-row.off circle, .lg-row.off rect, .lg-row.off polygon { opacity: .25; }

  /* ---------- section headers ---------- */
  .sec { display: flex; align-items: baseline; gap: 12px; margin: 34px 2px 14px; scroll-margin-top: 68px; }
  .sec h2 { font-size: 16px; font-weight: 650; margin: 0; letter-spacing: .01em; }
  .sec .count { color: var(--muted); font-size: 12.5px; }
  .sec .rule { flex: 1; border-top: 1px solid var(--line); transform: translateY(-4px); }
  .sec-btn { border: 1px solid var(--line); background: var(--card); border-radius: 999px;
             padding: 3px 12px; font-size: 12px; color: var(--muted); cursor: pointer; }
  .sec-btn:hover { color: var(--ink); border-color: var(--faint); }

  /* ---------- Table 1 (participant cohort) ---------- */
  #cohortbox { background: var(--card); border: 1px solid var(--line); border-radius: 14px;
               padding: 14px 18px 12px; overflow-x: auto; }
  .cohort-sub { font-size: 12px; color: var(--muted); margin: 0 0 10px; line-height: 1.45; }
  .cohort-sub b { color: var(--ink); font-weight: 650; }
  .cohort-tbl { width: 100%; border-collapse: collapse; font-size: 12.5px; line-height: 1.35; }
  .cohort-tbl th { text-align: left; font-size: 11px; font-weight: 700; letter-spacing: .05em;
                   text-transform: uppercase; color: var(--muted); padding: 7px 10px;
                   border-top: 1.5px solid var(--ink); border-bottom: 1px solid var(--ink);
                   background: #fafbfc; white-space: nowrap; }
  .cohort-tbl td { padding: 6px 10px; border-bottom: 1px solid var(--line); vertical-align: top;
                   color: var(--ink); white-space: nowrap; }
  .cohort-tbl tbody tr:last-child td { border-bottom: 1.5px solid var(--ink); }
  .cohort-row { cursor: pointer; transition: background .12s ease, opacity .15s ease; }
  .cohort-row:hover { background: #f4f7fb; }
  .cohort-row.selected { background: var(--accent-soft); }
  .cohort-row.selected td:first-child { color: var(--accent); font-weight: 700;
                                        box-shadow: inset 3px 0 0 var(--accent); }
  .cohort-row.off { opacity: .25; }
  .cohort-pid { font-weight: 650; color: var(--ink); }
  .cohort-sp { font-size: 11px; color: var(--muted); margin-left: 3px; }
  .cohort-tbl a { color: var(--accent); text-decoration: none; font-weight: 600; }
  .cohort-tbl a:hover { text-decoration: underline; }
  .cohort-foot { font-size: 11.5px; color: var(--muted); margin: 9px 0 0; line-height: 1.45; }
  .cohort-foot a { color: var(--accent); text-decoration: none; }
  .cohort-foot a:hover { text-decoration: underline; }

  /* ---------- spotlight ---------- */
  #spotlight { position: relative; border: 1.5px solid var(--accent); border-radius: 16px;
               background: #fdfeff; padding: 22px 26px 16px; margin: 26px 0 24px; scroll-margin-top: 68px; }
  #spotlight::before { content: "SELECTED STUDY"; position: absolute; top: -9px; left: 22px;
                       background: var(--accent); color: #fff; font-size: 9.5px; font-weight: 700;
                       letter-spacing: .12em; padding: 2px 10px; border-radius: 999px; }
  #sp-close { position: absolute; top: 10px; right: 12px; border: none; background: none;
              font-size: 22px; line-height: 1; color: var(--faint); cursor: pointer; }
  #sp-close:hover { color: var(--ink); }
  .sp-nav { display: flex; gap: 10px; align-items: center; margin-top: 14px;
            border-top: 1px dashed var(--line); padding-top: 10px; }
  .sp-nav button { border: 1px solid var(--line); background: var(--card); border-radius: 999px;
                   padding: 4px 14px; font-size: 12px; color: var(--muted); cursor: pointer; }
  .sp-nav button:hover:not(:disabled) { color: var(--accent); border-color: var(--accent); }
  .sp-nav button:disabled { opacity: .35; cursor: default; }
  #sp-pos { font-size: 12px; color: var(--faint); margin-left: auto; }

  /* ---------- study cards ---------- */
  #cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(310px, 1fr)); gap: 16px; }
  .cards-row-break { grid-column: 1 / -1; height: 0; border-top: 1px dashed var(--line); margin: 2px 0; }
  .study-card { --spine: #9aa3af; background: var(--card); border: 1px solid var(--line);
                border-left: 4px solid var(--spine); border-radius: 12px; padding: 14px 16px 12px;
                cursor: pointer; transition: box-shadow .15s ease, border-color .15s ease, transform .15s ease; }
  .study-card:hover { box-shadow: 0 4px 18px rgba(15, 36, 64, .10); transform: translateY(-1px); }
  .study-card.selected { border-color: var(--accent); border-left-color: var(--accent);
                         box-shadow: 0 0 0 2px var(--accent-soft); }
  .study-card.pinned { border-color: rgba(0, 107, 89, .38); border-left: 4px solid #006b59;
                       background: linear-gradient(180deg, rgba(0, 160, 135, .045) 0%, #ffffff 42%); }
  .sc-pin { display: inline-flex; align-items: center; gap: 3px; background: rgba(0, 160, 135, .13);
            color: #005f4f; font-size: 10.5px; font-weight: 700; letter-spacing: .04em;
            padding: 1px 7px; border-radius: 999px; white-space: nowrap; }
  .sc-head { display: flex; align-items: center; gap: 8px; }
  .sc-dot { width: 10px; height: 10px; border-radius: 50%; flex: none;
            box-shadow: 0 0 0 2px #fff, 0 0 0 3px rgba(0,0,0,.10); }
  .sc-title { font-size: 14.5px; font-weight: 650; }
  .sc-glyph { margin-left: auto; color: var(--muted); font-size: 12px; white-space: nowrap; }
  .sc-date { font-size: 11.5px; color: var(--faint); margin: 2px 0 1px; letter-spacing: .02em; }
  .sc-desc { font-size: 12.5px; color: var(--muted); margin: 2px 0 10px; }
  .sc-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 7px; }
  .sc-cell { background: #f8f9fb; border-radius: 9px; padding: 7px 10px 6px; }
  .sc-cell.span2 { grid-column: span 2; }
  .sc-cell .k { font-size: 9.5px; font-weight: 700; letter-spacing: .07em; color: var(--faint);
                text-transform: uppercase; margin-bottom: 2px; }
  .sc-cell .v { font-size: 13px; font-weight: 600; color: var(--ink); overflow-wrap: anywhere; }
  .sc-parts { font-size: 12.5px; color: var(--ink); margin-top: 9px; }
  .sc-parts .plabel { color: var(--faint); font-size: 11px; }
  .sc-links { margin-top: 8px; font-size: 12px; }
  .sc-links a { color: var(--accent); text-decoration: none; margin-right: 14px; font-weight: 600; }
  .sc-links a:hover { text-decoration: underline; }
  #cards-empty { display: none; color: var(--faint); font-size: 13px; padding: 26px 0;
                 text-align: center; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="head-row">
      <h1>__TITLE__</h1>
      <div class="head-tools">
        <span class="stat">Studies <b id="vis-count"></b></span>
        <span class="stat">Participants <b id="vis-people"></b></span>
        <input id="search" type="search" placeholder="Search author, participant, method…">
        <button id="reset">Reset</button>
      </div>
    </div>
  </header>

  <div id="panel">
    <div class="groups" id="controls"></div>
    <div class="bottom-row">
      <span class="mode-lbl">Layout</span>
      <div id="modebar">
        <button data-mode="dim" class="on">Dim</button>
        <button data-mode="reflow">Reflow</button>
        <button data-mode="hide">Hide in place</button>
      </div>
      <button id="pinned-only-btn" type="button" title="Toggle showing only the 3 open-data studies with vocabulary &gt; 71.4k and online WER &lt; 12.8%: Fan et al. (2023), Card et al. (2024) &amp; Jude et al. (2026a)">★ Open Data, Vocab &gt; 71.4k, WER &lt; 12.8%</button>
    </div>
  </div>

  <div class="sec" id="tl-sec" style="margin-top:20px">
    <h2>Timeline</h2>
    <span class="count" id="tl-count"></span>
    <span class="rule"></span>
    <button id="tl-toggle" class="sec-btn" type="button">Hide</button>
  </div>
  <div id="figbox">
    <button id="fs-btn" type="button" title="Toggle full screen (F)" aria-label="Toggle full screen">
      <svg id="fs-ico" viewBox="0 0 16 16"><path d="M2 6V2h4M14 6V2h-4M2 10v4h4M14 10v4h-4"/></svg>
      <span id="fs-lbl">Full screen</span>
    </button>
    <svg id="tl-svg" viewBox="0 0 __VW__ __VH__" xmlns="http://www.w3.org/2000/svg">__CHROME__</svg>
  </div>

  <div class="sec" id="cohort-sec">
    <h2>Participant Cohort</h2>
    <span class="count" id="cohort-count"></span>
    <span class="rule"></span>
    <button id="cohort-toggle" class="sec-btn" type="button">Show</button>
  </div>
  <div id="cohortbox" style="display:none">
    <p class="cohort-sub">
      <b>Ch. (impl.)</b> = total implanted channels · <b>Ch. (dec.)</b> = channels fed into the communication decoder ·
      <b>Interval</b> = symptom onset or diagnosis to implantation · <b>Home use</b> = independent daily use
    </p>
    <table class="cohort-tbl" aria-label="Table 1: The twenty participants of the communication-neuroprosthesis literature and company reports">
      <thead>
        <tr>
          <th>ID</th>
          <th>Institute</th>
          <th>Etiology</th>
          <th>Electrode</th>
          <th>Ch. (impl.)</th>
          <th>Ch. (dec.)</th>
          <th>Side</th>
          <th>Modality</th>
          <th>Interval</th>
          <th>Implanted</th>
          <th>Home use</th>
        </tr>
      </thead>
      <tbody>
        <tr class="cohort-row" data-pid="T5">
          <td class="cohort-pid">T5<span class="cohort-sp" title="Can speak naturally">▲</span></td>
          <td>BrainGate2</td><td>SCI</td><td>Utah</td><td>192</td><td>192</td><td>L</td>
          <td>handwriting</td><td>9 y</td><td>2016-08</td><td>—</td>
        </tr>
        <tr class="cohort-row" data-pid="T12">
          <td class="cohort-pid">T12<span class="cohort-sp" title="Vocal paralysis (dysarthria)">△</span></td>
          <td>BrainGate2</td><td>ALS</td><td>Utah</td><td>256</td><td>128<br>128</td><td>L</td>
          <td>speech<br>handwriting</td><td>9 y</td><td>2022-03</td><td>—</td>
        </tr>
        <tr class="cohort-row" data-pid="T15">
          <td class="cohort-pid">T15<span class="cohort-sp" title="Vocal paralysis (dysarthria)">△</span></td>
          <td>BrainGate2</td><td>ALS</td><td>Utah</td><td>256</td><td>256<br>256</td><td>L</td>
          <td>speech<br>cursor</td><td>5 y</td><td>2023-07</td>
          <td><a href="https://doi.org/10.1038/s41591-026-04414-6" target="_blank" rel="noopener" onclick="event.stopPropagation()">3,800 h, 444 d ↗</a></td>
        </tr>
        <tr class="cohort-row" data-pid="T16">
          <td class="cohort-pid">T16<span class="cohort-sp" title="Vocal paralysis (dysarthria)">△</span></td>
          <td>BrainGate2</td><td>stroke</td><td>Utah</td><td>256</td><td>64</td><td>L</td>
          <td>speech</td><td>19 y</td><td>2023-12</td><td>—</td>
        </tr>
        <tr class="cohort-row" data-pid="T17">
          <td class="cohort-pid">T17<span class="cohort-sp" title="Vocal paralysis (dysarthria / locked-in syndrome)">△</span></td>
          <td>BrainGate2</td><td>ALS</td><td>Utah</td><td>384</td><td>256<br>256</td><td>L</td>
          <td>speech<br>typing</td><td>3 y</td><td>2024-02</td><td>—</td>
        </tr>
        <tr class="cohort-row" data-pid="T18">
          <td class="cohort-pid">T18<span class="cohort-sp" title="Can speak naturally">▲</span></td>
          <td>BrainGate2</td><td>SCI</td><td>Utah</td><td>384</td><td>384<br>384</td><td>bi</td>
          <td>typing<br>handwriting</td><td>1.5 y</td><td>2024-10<sup>†</sup></td><td>—</td>
        </tr>
        <tr class="cohort-row" data-pid="T21">
          <td class="cohort-pid">T21<span class="cohort-sp" title="Vocal paralysis (dysarthria)">△</span></td>
          <td>BrainGate2</td><td>ALS</td><td>Utah</td><td>384</td><td>384</td><td>bi</td>
          <td>speech</td><td>3 y</td><td>&lt;2026-07</td><td>—</td>
        </tr>
        <tr class="cohort-row" data-pid="T22">
          <td class="cohort-pid">T22<span class="cohort-sp" title="Vocal paralysis (anarthria)">△</span></td>
          <td>BrainGate2</td><td>stroke</td><td>Utah</td><td>384</td><td>256</td><td>L</td>
          <td>speech</td><td>3 y</td><td>&lt;2026-07</td><td>—</td>
        </tr>
        <tr class="cohort-row" data-pid="B1">
          <td class="cohort-pid">B1<span class="cohort-sp" title="Vocal paralysis (anarthria)">△</span></td>
          <td>UCSF</td><td>stroke</td><td>ECoG</td><td>128<br>253</td><td>128<br>253</td><td>L</td>
          <td>speech<br>speech</td><td>16 y<br>21 y</td><td>2019-02<br>2024-04</td><td>—</td>
        </tr>
        <tr class="cohort-row" data-pid="B3">
          <td class="cohort-pid">B3<span class="cohort-sp" title="Vocal paralysis (anarthria)">△</span></td>
          <td>UCSF</td><td>stroke</td><td>ECoG</td><td>253</td><td>253</td><td>L</td>
          <td>speech</td><td>17 y</td><td>2022-09</td><td>—</td>
        </tr>
        <tr class="cohort-row" data-pid="B6">
          <td class="cohort-pid">B6<span class="cohort-sp" title="Vocal paralysis (severe vocal-tract paralysis)">△</span></td>
          <td>UCSF</td><td>ALS</td><td>ECoG</td><td>253</td><td>253<br>253</td><td>L</td>
          <td>speech<br>gesture</td><td>4 y</td><td>2025-03</td><td>—</td>
        </tr>
        <tr class="cohort-row" data-pid="CC01">
          <td class="cohort-pid">CC01<span class="cohort-sp" title="Vocal paralysis (dysarthria)">△</span></td>
          <td>JHU</td><td>ALS</td><td>ECoG</td><td>128</td><td>64</td><td>L</td>
          <td>speech</td><td>8 y</td><td>2022-07</td><td>—</td>
        </tr>
        <tr class="cohort-row" data-pid="P1">
          <td class="cohort-pid">P1<span class="cohort-sp" title="Can speak naturally">▲</span></td>
          <td>Caltech</td><td>SCI</td><td>Utah</td><td>288</td><td>96</td><td>L</td>
          <td>speech</td><td>1.5 y</td><td>2016-11</td><td>—</td>
        </tr>
        <tr class="cohort-row" data-pid="P2">
          <td class="cohort-pid">P2<span class="cohort-sp" title="Can speak naturally">▲</span></td>
          <td>Caltech</td><td>SCI</td><td>Utah</td><td>320</td><td>64</td><td>L</td>
          <td>speech</td><td>3 y</td><td>2022-10</td><td>—</td>
        </tr>
        <tr class="cohort-row" data-pid="Z1">
          <td class="cohort-pid">Z1<span class="cohort-sp" title="Can speak naturally">▲</span></td>
          <td>ZJU</td><td>SCI</td><td>Utah</td><td>192</td><td>96</td><td>L</td>
          <td>handwriting</td><td>~3 y</td><td>2019-08</td><td>—</td>
        </tr>
        <tr class="cohort-row" data-pid="V1">
          <td class="cohort-pid">V1<span class="cohort-sp" title="Vocal paralysis (dysarthria)">△</span></td>
          <td>Neuralink</td><td>ALS</td><td>N1</td><td>1,020</td><td>—</td><td>L</td>
          <td>speech</td><td>—</td><td>&lt;2026-10</td><td>—</td>
        </tr>
        <tr class="cohort-row" data-pid="V2">
          <td class="cohort-pid">V2<span class="cohort-sp" title="Vocal paralysis (dysarthria)">△</span></td>
          <td>Neuralink</td><td>ALS</td><td>N1</td><td>1,020</td><td>—</td><td>L</td>
          <td>speech</td><td>—</td><td>&lt;2026-10</td><td>—</td>
        </tr>
        <tr class="cohort-row" data-pid="V3">
          <td class="cohort-pid">V3<span class="cohort-sp" title="Vocal paralysis (dysarthria)">△</span></td>
          <td>Neuralink</td><td>ALS</td><td>N1</td><td>3,072</td><td>—</td><td>L</td>
          <td>speech</td><td>—</td><td>&lt;2026-10</td><td>—</td>
        </tr>
        <tr class="cohort-row" data-pid="P7">
          <td class="cohort-pid">P7<span class="cohort-sp" title="Can speak naturally">▲</span></td>
          <td>Neuralink</td><td>ALS</td><td>N1</td><td>—</td><td>—</td><td>—</td>
          <td>typing</td><td>—</td><td>2025-06<sup>‡</sup></td><td>—</td>
        </tr>
        <tr class="cohort-row" data-pid="C1">
          <td class="cohort-pid">C1<span class="cohort-sp" title="Vocal paralysis (dysarthria)">△</span></td>
          <td>Paradromics</td><td>MND</td><td>Connexus</td><td>—</td><td>—</td><td>—</td>
          <td>speech</td><td>—</td><td>2026-06<sup>‡</sup></td><td>—</td>
        </tr>
      </tbody>
    </table>
    <p class="cohort-foot">
      <b>▲</b> can speak naturally (6/20) · <b>△</b> vocal paralysis (14/20) · <b>Z1, V1–V3, C1</b> coined IDs ·
      <b>&lt;yyyy-mm</b> implanted before given month · <sup>†</sup>derived from post-implant day · <sup>‡</sup>company report
    </p>
  </div>

  <div class="sec" id="bench-sec">
    <h2>Vocabulary, Dataset Size, Word Error Rate &amp; Speed</h2>
    <span class="count" id="bench-count"></span>
    <span class="rule"></span>
    <button id="bench-toggle" class="sec-btn" type="button">Hide</button>
  </div>
  <div id="benchbox">
    <button id="f2-fs-btn" type="button" title="Toggle full screen" aria-label="Toggle full screen">
      <svg id="f2-fs-ico" viewBox="0 0 16 16"><path d="M2 6V2h4M14 6V2h-4M2 10v4h4M14 10v4h-4"/></svg>
      <span id="f2-fs-lbl">Full screen</span>
    </button>
    <svg id="f2-svg" viewBox="0 0 __F2_VW__ __F2_VH__" xmlns="http://www.w3.org/2000/svg">__FIG2_UPPER__</svg>
  </div>

  <div id="spotlight" style="display:none">
    <button id="sp-close" title="Close">×</button>
    <div id="sp-body"></div>
    <div class="sp-nav">
      <button id="sp-prev">← Previous</button>
      <button id="sp-next">Next →</button>
      <span id="sp-pos"></span>
    </div>
  </div>

  <div class="sec" id="cards-sec">
    <h2>Study Cards</h2>
    <span class="count" id="cards-count"></span>
    <span class="rule"></span>
    <button id="cards-toggle" class="sec-btn" type="button">Show</button>
  </div>
  <div id="cardsbox" style="display:none">
    <div id="cards"></div>
    <div id="cards-empty">No studies match the current filters.</div>
  </div>
</div>

<script>
'use strict';
const STUDIES = __STUDIES__;
const GEOM = __GEOM__;
const META = __META__;
const N = STUDIES.length;

const SVGNS = 'http://www.w3.org/2000/svg';
const PT = 1.3889;                       // svg units per pt (200 dpi, 0.5 scale)
const KX = (GEOM.sx1 - GEOM.sx0) / (GEOM.x1 - GEOM.x0);
const KY = (GEOM.sy0 - GEOM.sy1) / (GEOM.y1 - GEOM.y0);   // negative: data grows upward
const X = v => GEOM.sx0 + (v - GEOM.x0) * KX;
const Y = v => GEOM.sy1 + (v - GEOM.y0) * KY;
const COLORS = { open:'#00A087', partial:'#c07f00', closed:'#cc3324', unknown:'#7F7F7F', reanalysis:'#7B3294' };
const AVAIL_LBL = { open:'open data', partial:'data on request', closed:'closed data',
                    unknown:'preprint', reanalysis:'reanalysis on open data' };
const MODE_LBL = { online:'online', offline:'offline' };
const ELEC_LBL = { ecog:'ECoG', utah:'Utah array', thin_film:'thin-film', microwire:'microwire' };
const TASK_LBL = { inner:'inner speech ◆', intended:'overt/mouthing/intended ◇', both:'inner ◆ + overt ◇' };
const OUT_LBL = { voice:'brain-to-voice ⬟', text:'brain-to-text ⬠', both:'voice ⬟ + text ⬠' };
const SHAPE_GLYPH = { o:'●', s:'■', h:'⬢', v:'▼' };
const LANE_LBL = { speech:'Speech', typing:'Typing', handwriting:'Handwriting' };
const LANES = ['speech', 'typing', 'handwriting'];
const TRACKS = GEOM.tracks, ROWS = GEOM.rows;

function el(tag, attrs, parent){
  const e = document.createElementNS(SVGNS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
function esc(s){ return String(s).replace(/[&<>"']/g,
  c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

function markerPath(shape, r){
  if (shape === 'o')
    return `M ${-r} 0 A ${r} ${r} 0 1 0 ${r} 0 A ${r} ${r} 0 1 0 ${-r} 0 Z`;
  let p;
  if (shape === 's'){ const k = r * 0.92; p = [[k,k],[-k,k],[-k,-k],[k,-k]]; }
  else if (shape === 'h'){ p = []; for (let i=0;i<6;i++){ const a = Math.PI/2 + i*Math.PI/3; p.push([r*1.06*Math.cos(a), -r*1.06*Math.sin(a)]); } }
  else { p = [[0, r*1.12],[r*1.05, -r*0.78],[-r*1.05, -r*0.78]]; }
  return 'M ' + p.map(q => q[0].toFixed(2) + ' ' + q[1].toFixed(2)).join(' L ') + ' Z';
}

/* ---------- filter state ----------
   decoded_output and task_instruction intentionally expose exactly TWO options
   each (voice/text, inner/intended). Studies recorded as "both" remain visible
   while at least one of the two options is selected. */
const DIMS = {
  data_availability: ['open','partial','closed','unknown','reanalysis'],
  decoding_mode: ['online','offline'],
  decoded_output: ['voice','text'],
  electrode: ['ecog','utah','thin_film','microwire'],
  speech_ability: ['intact','paralysis'],
  task_instruction: ['inner','intended'],
  lane: ['speech','typing','handwriting'],
};
const STAR_STUDY_IDS = new Set(['fan2023', 'card2024', 'jude2026a']);
const sel = {}; for (const d in DIMS) sel[d] = new Set(DIMS[d]);
const ALL_DATES = STUDIES.map(r => r.date).sort();
const MIN_DATE = ALL_DATES[0] || '2021-01-01';
const MAX_DATE = ALL_DATES[ALL_DATES.length - 1] || '2026-12-31';
let searchStr = '', dateFrom = MIN_DATE, dateTo = MAX_DATE, mode = 'dim', selectedId = null, selectedPid = null, pinnedOnly = false;

function matches(rec){
  if (pinnedOnly && !STAR_STUDY_IDS.has(rec.id)) return false;
  if (!sel.data_availability.has(rec.opn)) return false;
  if (!sel.decoding_mode.has(rec.decoding_mode)) return false;
  if (!sel.electrode.has(rec.electrode)) return false;
  if (!sel.lane.has(rec.lane)) return false;
  const dsel = sel.decoded_output;
  if (rec.decoded_output === 'both' ? dsel.size === 0 : !dsel.has(rec.decoded_output)) return false;
  const tsel = sel.task_instruction;
  if (rec.task_instruction === 'both' ? tsel.size === 0 : !tsel.has(rec.task_instruction)) return false;
  if (rec.speech_abilities.length &&
      !rec.speech_abilities.some(a => sel.speech_ability.has(a))) return false;
  if (selectedPid && !rec.unique_pids.includes(selectedPid)) return false;
  const lo = dateFrom && dateTo && dateFrom > dateTo ? dateTo : dateFrom;
  const hi = dateFrom && dateTo && dateFrom > dateTo ? dateFrom : dateTo;
  if (lo && rec.date < lo) return false;
  if (hi && rec.date > hi) return false;
  if (searchStr){
    const hay = (rec.citation + ' ' + rec.id + ' ' + rec.participants.join(' ') + ' ' +
                 rec.descriptor + ' ' + (rec.metrics.sentences || '')).toLowerCase();
    if (!hay.includes(searchStr)) return false;
  }
  return true;
}

/* ---------- build SVG node groups ---------- */
const svg = document.getElementById('tl-svg');
const emptyMsg = document.getElementById('empty-msg');
const layerStems  = el('g', {}, null); svg.insertBefore(layerStems, emptyMsg);
const layerMarks  = el('g', {}, null); svg.insertBefore(layerMarks, emptyMsg);
const layerCards  = el('g', {}, null); svg.insertBefore(layerCards, emptyMsg);

const nodes = STUDIES.map(rec => {
  const gMark = el('g', {class:'mark'}, layerMarks);
  const r = (rec.decoding_mode === 'online' ? 11.5 : 8.3) * PT / 2;
  if (rec.decoding_mode === 'online')
    el('path', {d: markerPath(rec.shape, r), fill: COLORS[rec.opn], stroke: 'white',
                'stroke-width': 1.2 * PT}, gMark);
  else
    el('path', {d: markerPath(rec.shape, r), fill: 'white', stroke: COLORS[rec.opn],
                'stroke-width': 2.0 * PT}, gMark);
  const gStem = el('g', {}, layerStems);
  const gCard = el('g', {class:'card'}, layerCards);
  const t1 = el('text', {class:'t-name', 'font-size': (8*PT).toFixed(2), 'font-weight': 600,
                         fill:'#1f2937', 'text-anchor':'middle', 'dominant-baseline':'central'}, gCard);
  t1.textContent = rec.name_line;
  const t2 = el('text', {class:'t-metric', 'font-size': (7*PT).toFixed(2), fill:'#6b7280',
                         'text-anchor':'middle', 'dominant-baseline':'central'}, gCard);
  t2.textContent = rec.metric_line;
  const rv = {rec, gMark, gStem, gCard, t1, t2};
  const pick = ev => { ev.stopPropagation(); selectStudy(rec.id, true); };
  gCard.addEventListener('click', pick);
  gMark.addEventListener('click', pick);
  return rv;
});

/* ---------- JS port of the two-pass layout solver ---------- */
function resolveSide(lane, ord){
  if (lane === 'speech') return ord % 2 === 0 ? 'below' : 'above';
  if (lane === 'handwriting' && ord >= 2) return ord % 2 === 0 ? 'above' : 'below';
  return 'below';
}
function pack(placed, xm, hw, cap){
  let cx = Math.max(xm, GEOM.padx_l + hw);
  for (const [pcx, phw] of placed.slice().sort((a,b) => a[0]-b[0])){
    if (cx - hw < pcx + phw + GEOM.gap) cx = pcx + phw + hw + GEOM.gap;
  }
  if (cx - xm > cap || cx + hw > GEOM.padx_r) return null;
  return cx;
}
function computeLayout(vis){
  const placed = {};
  for (const lane of LANES){ placed[lane] = {};
    for (const s of ['above','below']) for (const y of ROWS[lane][s]) placed[lane][y] = []; }
  const LEG_X_END = GEOM.leg_x_end;
  const TOP_ROW = ROWS.speech.above[ROWS.speech.above.length - 1];
  placed.speech[TOP_ROW].push([0.5*(GEOM.x0 + LEG_X_END - GEOM.gap),
                               0.5*(LEG_X_END - GEOM.gap - GEOM.x0)]);
  const ordMap = {};
  for (const lane of LANES){
    vis.filter(n => n.rec.lane === lane)
       .sort((a,b) => a.rec.xm - b.rec.xm)
       .forEach((n,i) => ordMap[n.rec.id] = i);
  }
  const L = {};
  for (const n of vis){
    const rec = n.rec, hw = n.rec.hw;
    const pref = resolveSide(rec.lane, ordMap[rec.id]);
    const sides = rec.lane === 'speech' ? [pref] : [pref, pref === 'below' ? 'above' : 'below'];
    let got = null;
    for (const cap of [0.45, GEOM.maxshift]){
      for (const s of sides){
        for (const y of ROWS[rec.lane][s]){
          const cx = pack(placed[rec.lane][y], rec.xm, hw, cap);
          if (cx !== null && !(y === TOP_ROW && cx - hw < LEG_X_END)){ got = [cx, y, s]; break; }
        }
        if (got) break;
      }
      if (got) break;
    }
    if (!got){ console.warn('pack fallback for', rec.id);
      got = [rec.xm, ROWS[rec.lane][pref][0], pref]; }
    placed[rec.lane][got[1]].push([got[0], hw]);
    L[rec.id] = {cx: got[0], y: got[1], side: got[2]};
  }
  return L;
}
function computeMarks(vis, L){
  const MARK = {};
  for (const lane of LANES){
    for (const n of vis.filter(n => n.rec.lane === lane).sort((a,b) => a.rec.xm - b.rec.xm)){
      const ty = TRACKS[n.rec.lane], sgn = L[n.rec.id].side === 'above' ? 1 : -1;
      if (n.rec.opn === 'open') MARK[n.rec.id] = {x: n.rec.xm, y: ty};
      else {
        let lvl = 0.52;
        const vals = Object.values(MARK);
        while (vals.some(m => Math.abs(n.rec.xm - m.x) < 0.045 &&
                              Math.abs(ty + sgn*lvl - m.y) < 1e-9)) lvl += 0.52;
        MARK[n.rec.id] = {x: n.rec.xm, y: ty + sgn * lvl};
      }
    }
  }
  return MARK;
}
function drawAll(active, geomOf){
  for (const n of nodes) n.gStem.innerHTML = '';
  const byLaneSide = {};
  for (const n of active){
    const g = geomOf(n);
    const key = n.rec.lane + '|' + g.side;
    (byLaneSide[key] = byLaneSide[key] || []).push([n.rec.xm, n.rec.id]);
  }
  const stemX = {};
  for (const k in byLaneSide){
    const lst = byLaneSide[k].sort((a,b) => a[0] - b[0]);
    let i = 0;
    while (i < lst.length){
      const grp = [lst[i]];
      while (i + 1 < lst.length && Math.abs(lst[i+1][0] - lst[i][0]) < 0.045){ i++; grp.push(lst[i]); }
      if (grp.length === 1) stemX[grp[0][1]] = grp[0][0];
      else grp.forEach((g, gi) => { stemX[g[1]] = g[0] + (gi - 0.5*(grp.length-1)) * 0.022; });
      i++;
    }
  }
  const rowRails = {};
  const P = v => v.toFixed(2);
  for (const n of active){
    const rec = n.rec, g = geomOf(n);
    const mark = g.mark, sgn = g.side === 'above' ? 1 : -1;
    const xs = stemX[rec.id] !== undefined ? stemX[rec.id] : rec.xm;
    const yStart = mark.y + sgn * 0.17;
    const mk = pts => el('polyline', {
      points: pts.map(p => P(X(p[0])) + ',' + P(Y(p[1]))).join(' '),
      fill:'none', stroke:'#c3c9d1', 'stroke-width': 0.8 * PT }, n.gStem);
    if (Math.abs(g.cx - xs) > 0.02){
      const xLo = Math.min(xs, g.cx), xHi = Math.max(xs, g.cx);
      const rk = rec.lane + '|' + g.side + '|' + g.y;
      const rails = (rowRails[rk] = rowRails[rk] || []);
      const used = new Set(rails.filter(r => !(xHi < r[0]-0.04 || xLo > r[1]+0.04)).map(r => r[2]));
      let lvl = 0; while (used.has(lvl)) lvl++;
      rails.push([xLo, xHi, lvl]);
      const yRail = g.y - sgn * (0.22 * lvl);
      if (Math.abs(xs - rec.xm) > 1e-4) mk([[rec.xm, yStart],[xs, yStart + sgn*0.10],[xs, yRail]]);
      else mk([[xs, yStart],[xs, yRail]]);
      mk([[xs, yRail],[g.cx, yRail]]);
      if (lvl > 0) mk([[g.cx, yRail],[g.cx, g.y]]);
    } else {
      mk([[rec.xm, yStart],[g.cx, g.y]]);
    }
    n.gMark.setAttribute('transform', `translate(${P(X(mark.x))},${P(Y(mark.y))})`);
    const nameDy = g.side === 'above' ? 0.59 : -0.22;
    const metDy  = g.side === 'above' ? 0.22 : -0.59;
    n.t1.setAttribute('x', P(X(g.cx))); n.t1.setAttribute('y', P(Y(g.y + nameDy)));
    n.t2.setAttribute('x', P(X(g.cx))); n.t2.setAttribute('y', P(Y(g.y + metDy)));
  }
}

/* ---------- standalone study cards ---------- */
function cardHtml(rec){
  const m = rec.metrics;
  const isStar = STAR_STUDY_IDS.has(rec.id);
  const pinBadge = isStar ? `<span class="sc-pin" title="Pinned: low online WER (&lt;13%)">★ Pinned</span>` : '';
  const dataHref = rec.data_url || (rec.data_doi ? 'https://doi.org/' + encodeURIComponent(rec.data_doi) : '');
  const links = [
    rec.doi ? `<a href="https://doi.org/${encodeURIComponent(rec.doi)}" target="_blank" rel="noopener">Paper ↗</a>` : '',
    dataHref ? `<a href="${esc(dataHref)}" target="_blank" rel="noopener">Dataset ↗</a>` : '',
    rec.url ? `<a href="${rec.url}" target="_blank" rel="noopener">Announcement ↗</a>` : '',
  ].join('');
  return `
    <div class="sc-head"><span class="sc-dot" style="background:${COLORS[rec.opn]}"></span>
      <span class="sc-title">${esc(rec.citation)}</span>${pinBadge}
      <span class="sc-glyph">${SHAPE_GLYPH[rec.shape]} ${ELEC_LBL[rec.electrode]}</span></div>
    <div class="sc-date">${esc(rec.date)} · ${LANE_LBL[rec.lane]} lane · ${AVAIL_LBL[rec.opn]} · ${MODE_LBL[rec.decoding_mode]}</div>
    <div class="sc-desc">${esc(rec.descriptor)}</div>
    <div class="sc-grid">
      <div class="sc-cell"><div class="k">Vocabulary</div><div class="v">${esc(m.vocab_size || 'n/r')}</div></div>
      <div class="sc-cell"><div class="k">Speed</div><div class="v">${esc(m.speed || 'n/r')}</div></div>
      <div class="sc-cell"><div class="k">Error rate</div><div class="v">${esc(m.wer || 'n/r')}</div></div>
      <div class="sc-cell span2"><div class="k">Sentences / data volume</div><div class="v">${esc(m.sentences || 'n/r')}</div></div>
    </div>
    <div class="sc-parts"><span class="plabel">PARTICIPANTS (n=${rec.n_count})</span>&ensp;${rec.participants.map(esc).join(' · ')}</div>
    <div class="sc-parts"><span class="plabel">TASK</span>&ensp;${TASK_LBL[rec.task_instruction]}</div>
    <div class="sc-parts"><span class="plabel">OUTPUT</span>&ensp;${OUT_LBL[rec.decoded_output]}</div>
    <div class="sc-links">${links}</div>`;
}

const cardsBox = document.getElementById('cards');
function rebuildCards(visNodes){
  cardsBox.innerHTML = '';
  const pinnedNodes = visNodes.filter(n => STAR_STUDY_IDS.has(n.rec.id)).sort((a, b) => a.rec.xm - b.rec.xm);
  const restNodes = visNodes.filter(n => !STAR_STUDY_IDS.has(n.rec.id)).sort((a, b) => a.rec.xm - b.rec.xm);
  const appendCard = (n, isPinned) => {
    const d = document.createElement('div');
    d.className = 'study-card' + (isPinned ? ' pinned' : '');
    d.dataset.id = n.rec.id;
    d.style.setProperty('--spine', COLORS[n.rec.opn]);
    d.innerHTML = cardHtml(n.rec);
    d.addEventListener('click', () => selectStudy(n.rec.id, false));
    cardsBox.appendChild(d);
  };
  for (const n of pinnedNodes) appendCard(n, true);
  if (pinnedNodes.length && restNodes.length){
    const br = document.createElement('div');
    br.className = 'cards-row-break';
    cardsBox.appendChild(br);
  }
  for (const n of restNodes) appendCard(n, false);
  document.getElementById('cards-empty').style.display = visNodes.length ? 'none' : '';
  document.getElementById('cards-count').textContent =
    visNodes.length + ' of ' + N + ' studies';
}

/* ---------- spotlight ---------- */
function visibleNodes(){ return nodes.filter(n => matches(n.rec)); }
function navList(){ return visibleNodes().sort((a, b) => a.rec.xm - b.rec.xm); }
function selectStudy(id, scrollPage){
  selectedId = id;
  for (const n of nodes) n.gCard.classList.toggle('selected', n.rec.id === id);
  document.querySelectorAll('.study-card, .f2-slot').forEach(c =>
    c.classList.toggle('selected', c.dataset.id === id));
  const selRec = id ? STUDIES.find(r => r.id === id) : null;
  const selStudyPids = new Set(selRec ? selRec.unique_pids : []);
  document.querySelectorAll('.cohort-row').forEach(row => {
    const pid = row.dataset.pid;
    const on = !row.classList.contains('off');
    row.classList.toggle('selected', on && (pid === selectedPid || selStudyPids.has(pid)));
  });
  const sp = document.getElementById('spotlight');
  if (!id){ sp.style.display = 'none'; return; }
  const rec = STUDIES.find(r => r.id === id);
  document.getElementById('sp-body').innerHTML = cardHtml(rec);
  sp.style.display = '';
  updateSpotlightNav();
  if (scrollPage !== false){
    if (typeof isFs === 'function' && isFs()) exitFs();
    sp.scrollIntoView({behavior:'smooth', block:'nearest'});
  }
}
function updateSpotlightNav(){
  const list = navList().map(n => n.rec.id);
  const i = list.indexOf(selectedId);
  document.getElementById('sp-pos').textContent =
    (i >= 0 ? (i + 1) + ' of ' + list.length + ' shown' : 'filtered out') +
    ' · ' + (STUDIES.findIndex(r => r.id === selectedId) + 1) + ' of ' + N + ' total';
  document.getElementById('sp-prev').disabled = i <= 0;
  document.getElementById('sp-next').disabled = i < 0 || i >= list.length - 1;
}
function stepSpotlight(delta){
  const list = navList().map(n => n.rec.id), i = list.indexOf(selectedId);
  if (i >= 0 && i + delta >= 0 && i + delta < list.length)
    selectStudy(list[i + delta]);
}
document.getElementById('sp-close').addEventListener('click', () => selectStudy(null));
document.getElementById('sp-prev').addEventListener('click', () => stepSpotlight(-1));
document.getElementById('sp-next').addEventListener('click', () => stepSpotlight(1));
document.addEventListener('keydown', e => {
  const tag = (e.target && e.target.tagName) || '';
  if (tag === 'INPUT' || tag === 'TEXTAREA') return;
  if ((e.key === 'f' || e.key === 'F') && !e.ctrlKey && !e.metaKey && !e.altKey){
    e.preventDefault(); toggleFs(); return;
  }
  if (e.key === 'Escape' && figbox.classList.contains('is-fullscreen')){
    exitFs(); return;
  }
  if (!selectedId) return;
  if (e.key === 'Escape') selectStudy(null);
  else if (e.key === 'ArrowLeft') stepSpotlight(-1);
  else if (e.key === 'ArrowRight') stepSpotlight(1);
});

/* ---------- totals ---------- */
function pidMatchesSearch(pid, rawTok, q){
  return pid.toLowerCase().includes(q) || Boolean(rawTok && rawTok.toLowerCase().includes(q));
}
function pidActiveInStudy(pid, rec){
  if (selectedPid && pid !== selectedPid) return false;
  const idx = rec.unique_pids.indexOf(pid);
  if (idx < 0) return false;
  const rawTok = rec.participants[idx] || '';
  const ability = rawTok.endsWith('▲') ? 'intact' : 'paralysis';
  if (!sel.speech_ability.has(ability)) return false;
  if (searchStr){
    const anyPidMatch = rec.unique_pids.some((p, i) => pidMatchesSearch(p, rec.participants[i], searchStr));
    const studyMetaHay = (rec.citation + ' ' + rec.id + ' ' + rec.descriptor).toLowerCase();
    if (anyPidMatch && !studyMetaHay.includes(searchStr)){
      if (!pidMatchesSearch(pid, rawTok, searchStr)) return false;
    }
  }
  return true;
}
function updateTotals(visNodes){
  const all = new Set(), perLane = {speech:new Set(), typing:new Set(), handwriting:new Set()};
  for (const n of visNodes) for (const pid of n.rec.unique_pids){
    if (!pidActiveInStudy(pid, n.rec)) continue;
    all.add(pid); perLane[n.rec.lane].add(pid);
  }
  document.getElementById('vis-count').textContent = visNodes.length + ' / ' + N;
  const tlCountEl = document.getElementById('tl-count');
  if (tlCountEl) tlCountEl.textContent = visNodes.length + ' of ' + N + ' studies';
  document.getElementById('vis-people').textContent = all.size + ' / ' + META.total_unique;
  document.getElementById('total-note').textContent = visNodes.length
    ? all.size + ' unique participants in total; ' + META.home_summary
    : 'no studies shown';
  for (const lane of LANES)
    document.getElementById('lt-' + lane).textContent =
      'n = ' + perLane[lane].size + ' unique participants';
  const cohortCountEl = document.getElementById('cohort-count');
  if (cohortCountEl){
    cohortCountEl.textContent =
      all.size + ' of ' + META.total_unique + ' participants' +
      (selectedPid ? ' · filtered by ' + selectedPid : '');
  }
  const selRec = selectedId ? STUDIES.find(r => r.id === selectedId) : null;
  const selStudyPids = new Set(selRec ? selRec.unique_pids : []);
  document.querySelectorAll('.cohort-row').forEach(row => {
    const pid = row.dataset.pid;
    const on = all.has(pid);
    row.classList.toggle('off', !on);
    row.classList.toggle('selected', on && (pid === selectedPid || selStudyPids.has(pid)));
  });
  emptyMsg.style.display = visNodes.length ? 'none' : '';
}

/* ---------- main apply ---------- */
function apply(){
  const visNodes = visibleNodes();
  const visIds = new Set(visNodes.map(n => n.rec.id));
  const L = mode === 'reflow' ? computeLayout(visNodes) : null;
  const MARK = mode === 'reflow' ? computeMarks(visNodes, L) : null;
  const geomOf = n => L
    ? {cx: L[n.rec.id].cx, y: L[n.rec.id].y, side: L[n.rec.id].side, mark: MARK[n.rec.id]}
    : {cx: n.rec.static_cx, y: n.rec.static_y, side: n.rec.static_side,
       mark: {x: n.rec.static_mark_x, y: n.rec.static_mark_y}};
  const active = mode === 'dim' ? nodes : visNodes;
  for (const n of nodes){
    const on = visIds.has(n.rec.id);
    if (mode === 'dim'){
      n.gCard.style.display = n.gStem.style.display = n.gMark.style.display = '';
      const op = on ? '1' : '0.13';
      n.gCard.style.opacity = op; n.gStem.style.opacity = op; n.gMark.style.opacity = op;
    } else {
      n.gCard.style.display = n.gStem.style.display = n.gMark.style.display = on ? '' : 'none';
      n.gCard.style.opacity = n.gStem.style.opacity = n.gMark.style.opacity = '1';
    }
  }
  drawAll(active, geomOf);
  updateTotals(visNodes);
  rebuildCards(visNodes);
  let f2VisA = 0, f2TotA = 0;
  const f2OpenIds = new Set(), f2OpenVis = new Set();
  document.querySelectorAll('.f2-slot').forEach(slot => {
    const on = visIds.has(slot.dataset.id);
    slot.style.opacity = on ? '1' : '0.13';
    slot.classList.toggle('selected', Boolean(selectedId && slot.dataset.id === selectedId));
  });
  document.querySelectorAll('.f2-drop').forEach(dr => {
    const sid = dr.dataset.id;
    const on = visIds.has(sid);
    dr.style.opacity = on ? '1' : '0.13';
    f2OpenIds.add(sid);
    if (on) f2OpenVis.add(sid);
  });
  const f2SlotsAll = Array.from(document.querySelectorAll('#f2-svg .f2-slot'));
  f2TotA = 32;
  for (let i = 0; i < Math.min(32, f2SlotsAll.length); i++){
    if (visIds.has(f2SlotsAll[i].dataset.id)) f2VisA++;
  }
  const benchCountEl = document.getElementById('bench-count');
  if (benchCountEl){
    benchCountEl.textContent =
      f2VisA + ' of ' + f2TotA + ' English communication-neuroprostheses studies';
  }
  if (selectedId && !visIds.has(selectedId) && mode !== 'dim') updateSpotlightNav();
  if (selectedId){
    document.querySelectorAll('.study-card').forEach(c =>
      c.classList.toggle('selected', c.dataset.id === selectedId));
  }
}

/* ---------- control panel ---------- */
const GROUPS = [
  {dim:'data_availability', title:'Data availability', opts:[
    {v:'open', label:'open', color:COLORS.open},
    {v:'partial', label:'on request', color:COLORS.partial},
    {v:'closed', label:'closed', color:COLORS.closed},
    {v:'unknown', label:'preprint', color:COLORS.unknown},
    {v:'reanalysis', label:'reanalysis', color:COLORS.reanalysis}]},
  {dim:'decoding_mode', title:'Decoding mode', opts:[
    {v:'online', label:'online', glyph:'●'},
    {v:'offline', label:'offline', glyph:'○'}]},
  {dim:'decoded_output', title:'Decoded output', opts:[
    {v:'voice', label:'brain-to-voice', glyph:'⬟'},
    {v:'text', label:'brain-to-text', glyph:'⬠'}]},
  {dim:'task_instruction', title:'Task instruction', opts:[
    {v:'inner', label:'inner', glyph:'◆'},
    {v:'intended', label:'overt/mouthing/intended', glyph:'◇'}]},
  {dim:'electrode', title:'Electrode', opts:[
    {v:'ecog', label:'ECoG', glyph:'●'},
    {v:'utah', label:'Utah', glyph:'■'},
    {v:'thin_film', label:'thin-film', glyph:'⬢'},
    {v:'microwire', label:'microwire', glyph:'▼'}]},
  {dim:'speech_ability', title:'Speech ability', opts:[
    {v:'intact', label:'can speak naturally', glyph:'▲'},
    {v:'paralysis', label:'vocal paralysis', glyph:'△'}]},
  {dim:'lane', title:'Modality', opts:[
    {v:'speech', label:'speech'},
    {v:'typing', label:'typing'},
    {v:'handwriting', label:'handwriting'}]},
];
const controls = document.getElementById('controls');
for (const grp of GROUPS){
  const g = document.createElement('div'); g.className = 'fgroup';
  const t = document.createElement('div'); t.className = 'fg-title'; t.textContent = grp.title;
  g.appendChild(t);
  const chips = document.createElement('div'); chips.className = 'chips';
  for (const o of grp.opts){
    const chip = document.createElement('button');
    chip.type = 'button'; chip.className = 'chip on';
    chip.dataset.dim = grp.dim; chip.dataset.val = o.v;
    if (o.color){ const sw = document.createElement('span'); sw.className = 'sw';
      sw.style.background = o.color; chip.appendChild(sw); }
    if (o.glyph){ const mg = document.createElement('span'); mg.className = 'mg';
      mg.textContent = o.glyph; chip.appendChild(mg); }
    chip.appendChild(document.createTextNode(o.label));
    chip.addEventListener('click', () =>
      toggleVal(grp.dim, o.v, !sel[grp.dim].has(o.v)));
    chips.appendChild(chip);
  }
  g.appendChild(chips);
  controls.appendChild(g);
}

const dateGrp = document.createElement('div'); dateGrp.className = 'fgroup';
const dateTitle = document.createElement('div'); dateTitle.className = 'fg-title'; dateTitle.textContent = 'Time range';
dateGrp.appendChild(dateTitle);
const dateWrap = document.createElement('div'); dateWrap.className = 'date-range';
dateWrap.innerHTML =
  `<input id="date-from" type="date" aria-label="Start date" title="Start date" min="2021-01-01" max="2027-12-31" value="${MIN_DATE}">` +
  `<span class="date-sep">–</span>` +
  `<input id="date-to" type="date" aria-label="End date" title="End date" min="2021-01-01" max="2027-12-31" value="${MAX_DATE}">` +
  `<button id="date-clear" type="button" title="Reset time range" aria-label="Reset time range">×</button>`;
dateGrp.appendChild(dateWrap);
controls.appendChild(dateGrp);
const dateFromEl = document.getElementById('date-from');
const dateToEl = document.getElementById('date-to');
const dateClearEl = document.getElementById('date-clear');
function onDateChange(){
  dateFrom = dateFromEl.value || '';
  dateTo = dateToEl.value || '';
  syncUI(); apply();
}
dateFromEl.addEventListener('input', onDateChange);
dateFromEl.addEventListener('change', onDateChange);
dateToEl.addEventListener('input', onDateChange);
dateToEl.addEventListener('change', onDateChange);
dateClearEl.addEventListener('click', () => {
  dateFrom = MIN_DATE; dateTo = MAX_DATE;
  dateFromEl.value = MIN_DATE; dateToEl.value = MAX_DATE;
  syncUI(); apply();
});

function syncUI(){
  document.querySelectorAll('.chip[data-dim]').forEach(chip => {
    chip.classList.toggle('on', sel[chip.dataset.dim].has(chip.dataset.val));
  });
  document.querySelectorAll('.lg-row').forEach(row => {
    row.classList.toggle('off', !sel[row.dataset.dim].has(row.dataset.val));
  });
  const isDateFiltered = Boolean((dateFrom && dateFrom > MIN_DATE) || (dateTo && dateTo < MAX_DATE));
  dateWrap.classList.toggle('filtered', isDateFiltered);
  const pinBtn = document.getElementById('pinned-only-btn');
  if (pinBtn) pinBtn.classList.toggle('on', pinnedOnly);
}
document.getElementById('pinned-only-btn').addEventListener('click', () => {
  pinnedOnly = !pinnedOnly;
  syncUI(); apply();
});
function toggleVal(dim, val, on){
  if (on) sel[dim].add(val); else sel[dim].delete(val);
  syncUI(); apply();
}
document.querySelectorAll('.lg-row').forEach(row =>
  row.addEventListener('click', ev => {
    ev.stopPropagation();
    const dim = row.dataset.dim, val = row.dataset.val;
    toggleVal(dim, val, !sel[dim].has(val));
  }));
document.querySelectorAll('#modebar button').forEach(b =>
  b.addEventListener('click', () => {
    mode = b.dataset.mode;
    document.querySelectorAll('#modebar button').forEach(x =>
      x.classList.toggle('on', x === b));
    apply();
  }));
document.getElementById('reset').addEventListener('click', () => {
  for (const d in DIMS) sel[d] = new Set(DIMS[d]);
  searchStr = ''; document.getElementById('search').value = '';
  selectedPid = null;
  pinnedOnly = false;
  dateFrom = MIN_DATE; dateTo = MAX_DATE;
  dateFromEl.value = MIN_DATE; dateToEl.value = MAX_DATE;
  mode = 'dim';
  document.querySelectorAll('#modebar button').forEach(x =>
    x.classList.toggle('on', x.dataset.mode === 'dim'));
  selectStudy(null); syncUI(); apply();
});
/* ---------- full screen ---------- */
const figbox = document.getElementById('figbox');
const tlToggle = document.getElementById('tl-toggle');
const fsBtn = document.getElementById('fs-btn');
const fsLbl = document.getElementById('fs-lbl');
const fsIco = document.getElementById('fs-ico');
const cohortbox = document.getElementById('cohortbox');
const cohortToggle = document.getElementById('cohort-toggle');
const benchbox = document.getElementById('benchbox');
const f2FsBtn = document.getElementById('f2-fs-btn');
const f2FsLbl = document.getElementById('f2-fs-lbl');
const f2FsIco = document.getElementById('f2-fs-ico');
const benchToggle = document.getElementById('bench-toggle');
const cardsbox = document.getElementById('cardsbox');
const cardsToggle = document.getElementById('cards-toggle');
const FS_ENTER_PATH = 'M2 6V2h4M14 6V2h-4M2 10v4h4M14 10v4h-4';
const FS_EXIT_PATH  = 'M6 2v4H2M10 2v4h4M6 14v-4H2M10 14v-4h4';
function isBoxFs(box){
  return Boolean(document.fullscreenElement === box ||
                 document.webkitFullscreenElement === box ||
                 box.classList.contains('is-fullscreen'));
}
function isFs(){ return isBoxFs(figbox) || isBoxFs(benchbox); }
function syncFsUI(){
  const on1 = isBoxFs(figbox);
  fsLbl.textContent = on1 ? 'Exit full screen' : 'Full screen';
  fsIco.innerHTML = `<path d="${on1 ? FS_EXIT_PATH : FS_ENTER_PATH}"/>`;
  const on2 = isBoxFs(benchbox);
  f2FsLbl.textContent = on2 ? 'Exit full screen' : 'Full screen';
  f2FsIco.innerHTML = `<path d="${on2 ? FS_EXIT_PATH : FS_ENTER_PATH}"/>`;
  document.body.style.overflow =
    (figbox.classList.contains('is-fullscreen') || benchbox.classList.contains('is-fullscreen')) ? 'hidden' : '';
}
function exitFs(){
  if ((document.fullscreenElement === figbox || document.fullscreenElement === benchbox) && document.exitFullscreen)
    document.exitFullscreen().catch(() => {});
  else if ((document.webkitFullscreenElement === figbox || document.webkitFullscreenElement === benchbox) && document.webkitExitFullscreen)
    document.webkitExitFullscreen();
  figbox.classList.remove('is-fullscreen');
  benchbox.classList.remove('is-fullscreen');
  syncFsUI();
}
function toggleBoxFs(box){
  if (isBoxFs(box)){ exitFs(); return; }
  if (isFs()) exitFs();
  const req = box.requestFullscreen || box.webkitRequestFullscreen;
  if (req){
    try {
      const p = req.call(box);
      if (p && typeof p.catch === 'function')
        p.catch(() => { box.classList.add('is-fullscreen'); syncFsUI(); });
    } catch (_) {
      box.classList.add('is-fullscreen'); syncFsUI();
    }
  } else {
    box.classList.add('is-fullscreen'); syncFsUI();
  }
}
function toggleFs(){ toggleBoxFs(figbox); }
fsBtn.addEventListener('click', ev => { ev.stopPropagation(); toggleBoxFs(figbox); });
f2FsBtn.addEventListener('click', ev => { ev.stopPropagation(); toggleBoxFs(benchbox); });
document.addEventListener('fullscreenchange', syncFsUI);
document.addEventListener('webkitfullscreenchange', syncFsUI);

tlToggle.addEventListener('click', () => {
  const hidden = figbox.style.display === 'none';
  figbox.style.display = hidden ? '' : 'none';
  tlToggle.textContent = hidden ? 'Hide' : 'Show';
});

cohortToggle.addEventListener('click', () => {
  const hidden = cohortbox.style.display === 'none';
  cohortbox.style.display = hidden ? '' : 'none';
  cohortToggle.textContent = hidden ? 'Hide' : 'Show';
});

document.querySelectorAll('.cohort-row').forEach(row => {
  const pid = row.dataset.pid;
  row.addEventListener('click', () => {
    selectedPid = (selectedPid === pid) ? null : pid;
    apply();
  });
});

benchToggle.addEventListener('click', () => {
  const hidden = benchbox.style.display === 'none';
  benchbox.style.display = hidden ? '' : 'none';
  benchToggle.textContent = hidden ? 'Hide' : 'Show';
});

cardsToggle.addEventListener('click', () => {
  const hidden = cardsbox.style.display === 'none';
  cardsbox.style.display = hidden ? '' : 'none';
  cardsToggle.textContent = hidden ? 'Hide' : 'Show';
});

document.querySelectorAll('.f2-slot').forEach(slot => {
  const sid = slot.dataset.id;
  slot.addEventListener('click', ev => {
    ev.stopPropagation();
    selectStudy(sid, true);
  });
  slot.addEventListener('mouseenter', () => {
    document.querySelectorAll(`.f2-slot[data-id="${sid}"]`).forEach(s => s.classList.add('hovered'));
  });
  slot.addEventListener('mouseleave', () => {
    document.querySelectorAll(`.f2-slot[data-id="${sid}"]`).forEach(s => s.classList.remove('hovered'));
  });
});

let debounce = null;
document.getElementById('search').addEventListener('input', e => {
  clearTimeout(debounce);
  debounce = setTimeout(() => { searchStr = e.target.value.trim().toLowerCase(); apply(); }, 120);
});

syncUI();
apply();
</script>
</body>
</html>
"""

def export_interactive_html(
    out_html: str,
    studies: list[dict[str, Any]],
    geom: dict[str, Any],
    counts_summary: dict[str, Any],
    verify_overlaps: bool = True,
) -> None:
    """Write the single-file interactive SVG + HTML edition of the timeline.

    Reuses the Pass-2 measured card geometries for a pixel-identical initial view,
    then re-runs the same packing solver in JavaScript whenever the reader toggles
    the legend or sidebar filters, embeds the interactive SVG of Figure 2's upper
    half (Panels a-d) between the selected-study spotlight and the study cards,
    and renders one standalone study card per publication below.

    Parameters
    ----------
    out_html : str
        Destination path for the generated ``.html`` artifact.
    studies : list of dict of str to Any
        Structured study records produced by :func:`load_studies`, augmented with
        Pass-2 layout fields (``hw``, ``static_cx``, ``static_y``, ``static_side``,
        ``static_mark_x``, ``static_mark_y``).
    geom : dict of str to Any
        Data-to-SVG geometry payload captured from the Matplotlib figure.
    counts_summary : dict of str to Any
        Cohort summary containing ``total_unique``, ``home_summary``, and
        ``metadata`` used for the live participant counters.
    verify_overlaps : bool, default=True
        Whether to run the pairwise text bounding-box overlap check on Figure 2's
        upper half before emitting SVG.

    Returns
    -------
    None
        The output is written directly to ``out_html``.
    """
    import html as html_lib
    meta = counts_summary.get("metadata", {})
    meta_out = {
        "title": str(meta.get("title", "Living Timeline of Communication Neuroprostheses")),
        "total_unique": counts_summary.get("total_unique", 0),
        "home_summary": str(counts_summary.get("home_summary", "")),
    }
    f2_svg_inner, f2_vw, f2_vh = _build_fig2_upper_svg(studies, verify_overlaps=verify_overlaps)
    studies_json = json.dumps(studies, ensure_ascii=False).replace("</", "<\\/")
    doc = (_HTML_TEMPLATE
           .replace("__CHROME__", _build_chrome_svg(geom))
           .replace("__FIG2_UPPER__", f2_svg_inner)
           .replace("__F2_VW__", str(f2_vw))
           .replace("__F2_VH__", str(f2_vh))
           .replace("__STUDIES__", studies_json)
           .replace("__GEOM__", json.dumps(geom, ensure_ascii=False).replace("</", "<\\/"))
           .replace("__META__", json.dumps(meta_out, ensure_ascii=False).replace("</", "<\\/"))
           .replace("__TITLE__", html_lib.escape(meta_out["title"]))
           .replace("__VW__", f'{geom["sx1"] + 8:.0f}')
           .replace("__VH__", f'{geom["sy1"] + 8:.0f}'))
    with open(out_html, "w", encoding="utf-8", newline="\n") as f:
        f.write(doc)


def main() -> None:
    """Parse CLI arguments, render all timeline artifacts, and print the summary.

    Returns
    -------
    None
        Prints the rendered card count, cohort totals, and overlap-gate status
        to standard output.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--yaml", default=os.path.join(here, "studies.yaml"),
                    help="Path to studies.yaml")
    ap.add_argument("--out",
                    default=os.path.join(here, "communication_neuroprostheses_timeline"),
                    help="Output stem for .png and .pdf")
    ap.add_argument("--check-overlaps", action="store_true", default=True,
                    help="Run bounding-box overlap check before saving (default: True)")
    args = ap.parse_args()

    n_cards, summary = render_timeline(
        yaml_path=args.yaml,
        out_stem=args.out,
        verify_overlaps=args.check_overlaps,
    )
    print(f"[ok] Rendered {n_cards} cards -> {args.out}.{{png,pdf}}")
    print(f"[ok] Interactive edition -> {args.out}.html")
    print(f"     Cohort: {summary['total_unique']} unique participants "
          f"({summary['lane_totals']['speech']} speech, "
          f"{summary['lane_totals']['typing']} typing, "
          f"{summary['lane_totals']['handwriting']} handwriting)")
    print("     Overlap check: ALL CLEAN")


if __name__ == "__main__":
    main()
