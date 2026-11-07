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
import os
import sys

os.environ.setdefault("SOURCE_DATE_EPOCH", "946684800")  # deterministic PDF

import matplotlib
matplotlib.use("Agg")
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


def date_to_year_fraction(date_str):
    """Convert YYYY-MM-DD into a decimal year fraction."""
    d = dt.date.fromisoformat(str(date_str))
    doy = d.timetuple().tm_yday
    days_in_year = 366 if (d.year % 4 == 0 and (d.year % 100 != 0 or d.year % 400 == 0)) else 365
    return round(d.year + doy / days_in_year, 2)


def normalize_participant_id(token, aliases):
    """Strip ▲/△ speech-ability suffix and resolve re-implant aliases (e.g. B1r -> B1)."""
    raw = token.replace("\u25b2", "").replace("\u25b3", "").strip()
    return aliases.get(raw, raw)


def load_studies(yaml_path, sync_json=True):
    """Parse `studies.yaml` (or fallback `studies.json`) into `items` tuples
    and compute cohort summary counts."""
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


def check_figure_overlaps(fig):
    """Inspect all Text and Legend artists in `fig` and return any >1 px overlaps."""
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


def render_timeline(yaml_path="studies.yaml",
                    out_stem="communication_neuroprostheses_timeline",
                    verify_overlaps=True):
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

    def resolve_marks(out):
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
            per_level = {}
            for i, it in enumerate(items):
                if it[5] != lane:
                    continue
                level = round(MARK[i][1], 6)
                for (mx, _my) in per_level.setdefault(level, []):
                    assert abs(mx - MARK[i][0]) >= 0.045, (it[0], lane)
                per_level[level].append(MARK[i])
        return MARK

    def pack(placed, xm, hw, cap):
        cx = max(xm, PADX_L + hw)
        for pcx, phw in sorted(placed):
            if cx - hw < pcx + phw + GAP:
                cx = pcx + phw + hw + GAP
        if cx - xm > cap or cx + hw > PADX_R:
            return None
        return cx

    def layout(hws):
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

    def draw(out):
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
                 label="overt / mouthing / intended")]
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
    )
    return len(items), counts_summary


def _poly_pts(cx, cy, r, n, rot_deg=90.0):
    """SVG polygon points string for a regular n-gon of radius r centered at (cx, cy)."""
    import math
    pts = []
    for i in range(n):
        a = math.radians(rot_deg + i * 360.0 / n)
        pts.append(f"{cx + r * math.cos(a):.2f},{cy + r * math.sin(a):.2f}")
    return " ".join(pts)


def _build_chrome_svg(geom):
    """Static SVG chrome (axis, tracks, clickable legends, notes) in 0.5-scale px coords."""
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
             f'x2="{X(geom["vis_end"]):.1f}" y2="{Y(geom["yax"]):.1f}" stroke="#3a3a3a" stroke-width="{SW(1.2)}"/>')
    for yr in range(2021, geom["max_yr_tick"]):
        s.append(f'<line x1="{X(yr):.1f}" y1="{Y(geom["yax"]):.1f}" '
                 f'x2="{X(yr):.1f}" y2="{Y(geom["yax"] - 0.15):.1f}" stroke="#3a3a3a" stroke-width="{SW(1.2)}"/>')
        s.append(f'<text x="{X(yr):.1f}" y="{Y(-3.57):.1f}" font-size="{10 * PT:.1f}" fill="#3a3a3a" '
                 f'text-anchor="middle" dominant-baseline="central">{yr}</text>')

    # Lane tracks
    lane_label = {"speech": "Speech", "typing": "Typing", "handwriting": "Handwriting"}
    for lane, ty in geom["tracks"].items():
        s.append(f'<line x1="{X(geom["x0"]):.1f}" y1="{Y(ty):.1f}" x2="{X(geom["x1"] + 0.6):.1f}" '
                 f'y2="{Y(ty):.1f}" stroke="#d3d3d3" stroke-width="{SW(1.3)}"/>')
        s.append(f'<text x="{X(geom["x0"] - 0.18):.1f}" y="{Y(ty):.1f}" font-size="{11.5 * PT:.1f}" '
                 f'font-weight="bold" fill="#404040" text-anchor="end" dominant-baseline="central">'
                 f'{lane_label[lane]}</text>')

    # Legend boxes (clickable rows mirror the 6 figure legends)
    def marker_svg(kind, color, cx, cy):
        if kind == "solid-circle":
            return (f'<circle cx="{cx}" cy="{cy}" r="5" fill="{color}" stroke="white" stroke-width="1"/>')
        if kind == "hollow-circle":
            return (f'<circle cx="{cx}" cy="{cy}" r="3.6" fill="white" stroke="{color}" stroke-width="1.8"/>')
        if kind == "solid-square":
            return (f'<rect x="{cx - 4}" y="{cy - 4}" width="8" height="8" fill="{color}" '
                    f'stroke="white" stroke-width="1"/>')
        if kind == "solid-hex":
            return f'<polygon points="{_poly_pts(cx, cy, 5, 6, 0)}" fill="{color}" stroke="white" stroke-width="1"/>'
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
            ("hollow-diamond", "#1a1a1a", "overt / mouthing / intended", "intended"),
        ]),
        ("decoded_output", "decoded output", [
            ("solid-pentagon", "#1a1a1a", "brain-to-voice", "voice"),
            ("hollow-pentagon", "#1a1a1a", "brain-to-text", "text"),
        ]),
    ]

    for (dim, title, rows), box in zip(LEG_DEFS, geom["legend_boxes"]):
        bx, by, bw, bh = box["x"], box["y"], box["w"], box["h"]
        g = [f'<g class="legend" data-dim="{dim}">',
             f'<rect class="frame" x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{bh:.1f}" '
             f'fill="white" fill-opacity="0.95" stroke="#cccccc"/>',
             f'<text class="lg-title" x="{bx + 8:.1f}" y="{by + 13:.1f}" font-size="{8.2 * PT:.1f}" '
             f'font-weight="bold" fill="#303030">{title}</text>']
        ry = by + 28
        for kind, color, label, val in rows:
            g.append(f'<g class="lg-row" data-dim="{dim}" data-val="{val}">')
            g.append(marker_svg(kind, color, bx + 13, ry))
            g.append(f'<text x="{bx + 24:.1f}" y="{ry:.1f}" font-size="{7.8 * PT:.1f}" fill="#303030" '
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
             f'<text x="{mid_x:.0f}" y="{mid_y:.0f}" font-size="15" fill="#666666" '
             f'text-anchor="middle">No study cards match the current filters · 当前筛选条件下没有匹配的研究</text></g>')
    return "\n".join(s)


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ — Interactive Timeline</title>
<style>
  :root { --accent: #0053b3; }
  * { box-sizing: border-box; }
  body { margin: 0; background: #eef1f4; color: #1a1a1a;
         font-family: "DejaVu Sans", "Segoe UI", Arial, "PingFang SC", "Microsoft YaHei", sans-serif; }
  .wrap { max-width: 1560px; margin: 0 auto; padding: 20px 22px 60px; }
  h1 { font-size: 21px; margin: 4px 0 2px; }
  .sub { color: #667; font-size: 13px; margin-bottom: 16px; }
  .box { background: #fff; border-radius: 12px; padding: 14px 18px;
         box-shadow: 0 1px 3px rgba(0,0,0,.09); margin-bottom: 16px; }

  /* ---- control panel ---- */
  .status-row { display: flex; flex-wrap: wrap; gap: 14px; align-items: center;
                padding-bottom: 10px; border-bottom: 1px solid #e5e8ec; margin-bottom: 12px; }
  .status-row b { color: var(--accent); }
  .controls { display: flex; flex-wrap: wrap; gap: 6px 26px; align-items: flex-start; }
  .fgroup { min-width: 148px; }
  .fg-title { font-size: 12px; font-weight: 700; color: #445; margin: 4px 0 4px;
              display: flex; align-items: center; gap: 6px; }
  .allbtn { font-size: 10px; border: 1px solid #ccd; background: #f5f7fa; border-radius: 999px;
            padding: 1px 7px; cursor: pointer; color: #567; }
  .allbtn:hover { background: #e8edf5; }
  .fgroup label { display: flex; align-items: center; gap: 6px; font-size: 12.5px;
                  color: #334; margin: 2px 0; cursor: pointer; user-select: none; }
  .fgroup input[type=checkbox] { accent-color: var(--accent); }
  .sw { display: inline-block; width: 10px; height: 10px; border-radius: 50%; flex: none; }
  .mg { display: inline-block; width: 14px; text-align: center; flex: none; font-size: 12px; }
  #search { padding: 6px 10px; border: 1px solid #ccd; border-radius: 8px; font-size: 13px;
            width: 250px; }
  .modepick { font-size: 12.5px; color: #334; display: flex; gap: 4px; align-items: center; }
  .modepick label { cursor: pointer; padding: 2px 6px; border-radius: 6px; }
  .modepick input { accent-color: var(--accent); }
  #reset { padding: 6px 14px; border: 1px solid #ccd; background: #f5f7fa; border-radius: 8px;
           font-size: 12.5px; cursor: pointer; color: #345; }
  #reset:hover { background: #e8edf5; }

  /* ---- figure ---- */
  .fig-hint { font-size: 12.5px; color: #778; margin: 2px 0 8px; }
  #tl-svg { width: 100%; height: auto; display: block; }
  #tl-svg text { font-family: "DejaVu Sans", Arial, sans-serif; }
  .card { cursor: pointer; }
  .card .t-name { paint-order: stroke; stroke: rgba(255,255,255,.92); stroke-width: 3px; }
  .card.selected .t-name { fill: var(--accent); text-decoration: underline; }
  .mark { cursor: pointer; }
  .legend { cursor: pointer; }
  .legend:hover .frame { stroke: #909090; }
  .legend.off { opacity: .55; }
  .legend.off .frame { stroke-dasharray: 4 3; }
  .lg-row.off text { fill: #aab; }
  .lg-row.off circle, .lg-row.off rect, .lg-row.off polygon { opacity: .3; }

  /* ---- spotlight & cards ---- */
  #spotlight { position: relative; border: 2px solid var(--accent); border-radius: 14px;
               background: #fbfcff; padding: 18px 20px 12px; margin: 20px 0 8px; max-width: 700px; }
  #sp-close { position: absolute; top: 8px; right: 10px; border: none; background: none;
              font-size: 20px; color: #889; cursor: pointer; }
  #sp-close:hover { color: #234; }
  .sp-nav { display: flex; gap: 10px; align-items: center; margin-top: 10px; }
  .sp-nav button { border: 1px solid #cdd; background: #f5f7fa; border-radius: 8px;
                   padding: 5px 12px; font-size: 12.5px; cursor: pointer; }
  .sp-nav button:hover { background: #e8edf5; }
  #sp-pos { font-size: 12.5px; color: #667; }
  .cards-header { font-size: 15px; font-weight: 700; color: #334; margin: 26px 0 10px; }
  #cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 14px; }
  .study-card { border: 1px solid #dde; border-radius: 12px; padding: 12px 14px; background: #fff;
                cursor: pointer; transition: box-shadow .15s, border-color .15s; }
  .study-card:hover { box-shadow: 0 2px 10px rgba(20,40,90,.13); }
  .study-card.selected { border-color: var(--accent); box-shadow: 0 0 0 2px rgba(0,83,179,.22); }
  .sc-head { display: flex; align-items: center; gap: 7px; }
  .sc-dot { width: 11px; height: 11px; border-radius: 50%; flex: none; border: 1.5px solid #fff;
            box-shadow: 0 0 0 1px rgba(0,0,0,.15); }
  .sc-title { font-weight: 700; font-size: 14px; }
  .sc-shape { margin-left: auto; color: #556; font-size: 13px; }
  .sc-date { font-size: 12px; color: #778; margin: 3px 0 1px; }
  .sc-desc { font-size: 12.5px; color: #445; margin-bottom: 8px; }
  .sc-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 8px 0; }
  .sc-cell { background: #f6f8fb; border-radius: 9px; padding: 7px 9px; }
  .sc-cell.span2 { grid-column: span 2; }
  .sc-cell .k { font-size: 10.5px; color: #889; letter-spacing: .02em; margin-bottom: 2px; }
  .sc-cell .v { font-size: 13px; font-weight: 600; color: #223; overflow-wrap: anywhere; }
  .sc-chips { margin: 6px 0 4px; }
  .chip { display: inline-block; font-size: 11px; border: 1px solid #d4d9e0; border-radius: 999px;
          padding: 1.5px 8px; margin: 2px 4px 2px 0; color: #445; background: #fafbfd; }
  .chip .cdot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
                margin-right: 4px; vertical-align: -0.5px; }
  .sc-parts { font-size: 12.5px; color: #334; margin-top: 4px; }
  .sc-links { margin-top: 7px; font-size: 12px; }
  .sc-links a { color: var(--accent); text-decoration: none; margin-right: 12px; }
  .sc-links a:hover { text-decoration: underline; }
</style>
</head>
<body>
<div class="wrap">
  <h1>__TITLE__</h1>
  <div class="sub">Interactive edition · 交互版 — hover a card for a quick summary, click it for the full
    standalone study card; toggle figure legends or the filters below to choose which studies are shown.</div>

  <div class="box" id="panel">
    <div class="status-row">
      <span>Showing 展示: <b id="vis-count"></b></span>
      <span>Unique participants 被试: <b id="vis-people"></b></span>
      <span class="modepick">Layout 排版:
        <label><input type="radio" name="mode" value="reflow" checked>reflow 重排</label>
        <label><input type="radio" name="mode" value="hide">hide in place 原位隐藏</label>
        <label><input type="radio" name="mode" value="dim">dim 淡出</label>
      </span>
      <input id="search" type="search" placeholder="Search 搜索: author / participant / descriptor…">
      <button id="reset">Reset 重置</button>
    </div>
    <div class="controls" id="controls"></div>
  </div>

  <div class="box">
    <div class="fig-hint">Tip 提示: click a legend entry above a lane track to toggle a class (e.g. the purple
      reanalysis or gray preprint markers); click any small timeline card or marker to open its standalone
      study card below.</div>
    <svg id="tl-svg" viewBox="0 0 __VW__ __VH__" xmlns="http://www.w3.org/2000/svg">__CHROME__</svg>
  </div>

  <div id="spotlight" style="display:none">
    <button id="sp-close" title="Close 关闭">×</button>
    <div id="sp-body"></div>
    <div class="sp-nav">
      <button id="sp-prev">◀ Prev 上一篇</button>
      <span id="sp-pos"></span>
      <button id="sp-next">Next 下一篇 ▶</button>
    </div>
  </div>

  <div class="cards-header">Standalone study cards · 研究卡片 (<span id="cards-count"></span>)</div>
  <div id="cards"></div>
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
const AVAIL_LBL = { open:'open data 开源', partial:'data on request 按需', closed:'closed data 未公开',
                    unknown:'preprint 预印本', reanalysis:'reanalysis 再分析' };
const MODE_LBL = { online:'online 实时闭环', offline:'offline 离线分析' };
const ELEC_LBL = { ecog:'ECoG', utah:'Utah array', thin_film:'thin-film', microwire:'microwire' };
const ELEC_CHAR = { ecog:'●', utah:'■', thin_film:'⬢', microwire:'▼' };
const TASK_LBL = { inner:'inner speech 内隐', intended:'overt / intended 外显', both:'inner+overt 两者' };
const OUT_LBL = { voice:'brain-to-voice B2V', text:'brain-to-text B2T', both:'B2V + B2T' };
const OUT_CHAR = { voice:'⬟', text:'⬠', both:'⬟/⬠' };
const SHAPE_CHAR = { o:'●', s:'■', h:'⬢', v:'▼' };
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
  else if (shape === 'h'){ p = []; for (let i=0;i<6;i++){ const a = Math.PI/i*0 + i*Math.PI/3; p.push([r*1.06*Math.cos(a), r*1.06*Math.sin(a)]); } }
  else { p = [[0,-r*1.12],[r*1.05, r*0.78],[-r*1.05, r*0.78]]; }
  return 'M ' + p.map(q => q[0].toFixed(2) + ' ' + q[1].toFixed(2)).join(' L ') + ' Z';
}

/* ---------- filter state ---------- */
const DIMS = {
  data_availability: ['open','partial','closed','unknown','reanalysis'],
  decoded_output: ['voice','text','both'],
  decoding_mode: ['online','offline'],
  electrode: ['ecog','utah','thin_film','microwire'],
  speech_ability: ['intact','paralysis'],
  task_instruction: ['inner','intended','both'],
  lane: ['speech','typing','handwriting'],
};
const sel = {}; for (const d in DIMS) sel[d] = new Set(DIMS[d]);
let searchStr = '', mode = 'reflow', selectedId = null;

function matches(rec){
  if (!sel.data_availability.has(rec.opn)) return false;
  if (!sel.decoded_output.has(rec.decoded_output)) return false;
  if (!sel.decoding_mode.has(rec.decoding_mode)) return false;
  if (!sel.electrode.has(rec.electrode)) return false;
  if (!sel.lane.has(rec.lane)) return false;
  if (!sel.task_instruction.has(rec.task_instruction)) return false;
  if (rec.speech_abilities.length &&
      !rec.speech_abilities.some(a => sel.speech_ability.has(a))) return false;
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
  const t1 = el('text', {class:'t-name', 'font-size': (8*PT).toFixed(2), 'font-weight': 700,
                         fill:'#1a1a1a', 'text-anchor':'middle', 'dominant-baseline':'central'}, gCard);
  t1.textContent = rec.name_line;
  const t2 = el('text', {class:'t-metric', 'font-size': (7*PT).toFixed(2), fill:'#555555',
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
      fill:'none', stroke:'#b8b8b8', 'stroke-width': 0.8 * PT }, n.gStem);
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
  const chips = [
    `<span class="chip"><span class="cdot" style="background:${COLORS[rec.opn]}"></span>${AVAIL_LBL[rec.opn] || rec.data_availability}</span>`,
    `<span class="chip">${MODE_LBL[rec.decoding_mode]}</span>`,
    `<span class="chip">${ELEC_CHAR[rec.shape]} ${ELEC_LBL[rec.electrode]}</span>`,
    `<span class="chip">${TASK_LBL[rec.task_instruction]}</span>`,
    `<span class="chip">${OUT_CHAR[rec.decoded_output]} ${OUT_LBL[rec.decoded_output]}</span>`,
    `<span class="chip">${LANE_LBL[rec.lane]}</span>`,
    rec.speech_abilities.includes('intact') ? '<span class="chip">▲ can speak naturally</span>' : '',
    rec.speech_abilities.includes('paralysis') ? '<span class="chip">△ vocal paralysis</span>' : '',
  ].join('');
  const links = [
    rec.doi ? `<a href="https://doi.org/${encodeURIComponent(rec.doi)}" target="_blank" rel="noopener">Paper DOI ↗</a>` : '',
    rec.data_doi ? `<a href="https://doi.org/${encodeURIComponent(rec.data_doi)}" target="_blank" rel="noopener">Dataset DOI ↗</a>` : '',
    rec.url ? `<a href="${rec.url}" target="_blank" rel="noopener">Announcement ↗</a>` : '',
  ].join('');
  return `
    <div class="sc-head"><span class="sc-dot" style="background:${COLORS[rec.opn]}"></span>
      <span class="sc-title">${esc(rec.citation)}</span><span class="sc-shape">${ELEC_CHAR[rec.shape]}</span></div>
    <div class="sc-date">${esc(rec.date)} · ${LANE_LBL[rec.lane]} lane</div>
    <div class="sc-desc">${esc(rec.descriptor)}</div>
    <div class="sc-grid">
      <div class="sc-cell"><div class="k">VOCABULARY 词库大小</div><div class="v">${esc(m.vocab_size || 'n/r')}</div></div>
      <div class="sc-cell"><div class="k">SPEED 速度</div><div class="v">${esc(m.speed || 'n/r')}</div></div>
      <div class="sc-cell"><div class="k">ERROR RATE 错误率</div><div class="v">${esc(m.wer || 'n/r')}</div></div>
      <div class="sc-cell span2"><div class="k">SENTENCES 数据量（句子）</div><div class="v">${esc(m.sentences || 'n/r')}</div></div>
    </div>
    <div class="sc-chips">${chips}</div>
    <div class="sc-parts">n = ${rec.n_count} · ${rec.participants.map(esc).join(' / ')}</div>
    <div class="sc-links">${links}</div>`;
}

const cardsBox = document.getElementById('cards');
function rebuildCards(visNodes){
  cardsBox.innerHTML = '';
  for (const n of visNodes.slice().sort((a, b) => a.rec.xm - b.rec.xm)){
    const d = document.createElement('div');
    d.className = 'study-card'; d.dataset.id = n.rec.id;
    d.innerHTML = cardHtml(n.rec);
    d.addEventListener('click', () => selectStudy(n.rec.id, false));
    d.addEventListener('click', () => {
      const sp = document.getElementById('spotlight');
      sp.scrollIntoView({behavior:'smooth', block:'nearest'});
    });
    cardsBox.appendChild(d);
  }
  document.getElementById('cards-count').textContent =
    visNodes.length + ' / ' + N;
}

/* ---------- spotlight ---------- */
function visibleNodes(){ return nodes.filter(n => matches(n.rec)); }
function navList(){ return visibleNodes().sort((a, b) => a.rec.xm - b.rec.xm); }
function selectStudy(id, scrollPage){
  selectedId = id;
  for (const n of nodes) n.gCard.classList.toggle('selected', n.rec.id === id);
  document.querySelectorAll('.study-card').forEach(c =>
    c.classList.toggle('selected', c.dataset.id === id));
  const sp = document.getElementById('spotlight');
  if (!id){ sp.style.display = 'none'; return; }
  const rec = STUDIES.find(r => r.id === id);
  document.getElementById('sp-body').innerHTML = cardHtml(rec);
  sp.style.display = '';
  updateSpotlightNav();
  if (scrollPage !== false) sp.scrollIntoView({behavior:'smooth', block:'nearest'});
}
function updateSpotlightNav(){
  const list = navList().map(n => n.rec.id);
  const i = list.indexOf(selectedId);
  document.getElementById('sp-pos').textContent =
    (i >= 0 ? (i + 1) + ' / ' + list.length + ' shown' : 'filtered out') +
    ' · ' + (STUDIES.findIndex(r => r.id === selectedId) + 1) + ' / ' + N + ' total';
  document.getElementById('sp-prev').disabled = i <= 0;
  document.getElementById('sp-next').disabled = i < 0 || i >= list.length - 1;
}
document.getElementById('sp-close').addEventListener('click', () => selectStudy(null));
document.getElementById('sp-prev').addEventListener('click', () => {
  const list = navList().map(n => n.rec.id), i = list.indexOf(selectedId);
  if (i > 0) selectStudy(list[i - 1]);
});
document.getElementById('sp-next').addEventListener('click', () => {
  const list = navList().map(n => n.rec.id), i = list.indexOf(selectedId);
  if (i >= 0 && i < list.length - 1) selectStudy(list[i + 1]);
});

/* ---------- totals ---------- */
function updateTotals(visNodes){
  const all = new Set(), perLane = {speech:new Set(), typing:new Set(), handwriting:new Set()};
  for (const n of visNodes) for (const pid of n.rec.unique_pids){
    all.add(pid); perLane[n.rec.lane].add(pid);
  }
  document.getElementById('vis-count').textContent = visNodes.length + ' / ' + N + ' studies';
  document.getElementById('vis-people').textContent = all.size + ' / ' + META.total_unique;
  document.getElementById('total-note').textContent = visNodes.length
    ? all.size + ' unique participants in total; ' + META.home_summary
    : 'no studies shown';
  for (const lane of LANES)
    document.getElementById('lt-' + lane).textContent =
      'n = ' + perLane[lane].size + ' unique participants';
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
  if (selectedId && !visIds.has(selectedId) && mode !== 'dim') updateSpotlightNav();
  if (selectedId){
    document.querySelectorAll('.study-card').forEach(c =>
      c.classList.toggle('selected', c.dataset.id === selectedId));
  }
}

/* ---------- control panel ---------- */
const GROUP_TITLES = {
  data_availability: 'Data availability 数据状态', decoded_output: 'Decoded output 解码输出',
  decoding_mode: 'Decoding mode 解码模式', electrode: 'Electrode 电极',
  speech_ability: 'Speech ability 言语能力', task_instruction: 'Task instruction 任务范式',
  lane: 'Modality lane 模态',
};
const GROUP_META = {
  data_availability: {sw: v => COLORS[v] === '#7F7F7F' ? '#7F7F7F' : COLORS[v], lbl: AVAIL_LBL},
  decoded_output: {mg: v => OUT_CHAR[v], lbl: OUT_LBL},
  decoding_mode: {mg: () => '●', lbl: MODE_LBL},
  electrode: {mg: v => ELEC_CHAR[v === 'ecog' ? 'o' : v === 'utah' ? 's' : v === 'thin_film' ? 'h' : 'v'], lbl: ELEC_LBL},
  speech_ability: {mg: v => v === 'intact' ? '▲' : '△',
                   lbl: {intact: 'can speak naturally 有自然言语', paralysis: 'vocal paralysis 言语瘫痪'}},
  task_instruction: {mg: v => v === 'inner' ? '◆' : v === 'intended' ? '◇' : '◆/◇', lbl: TASK_LBL},
  lane: {mg: () => '—', lbl: {speech:'Speech', typing:'Typing', handwriting:'Handwriting'}},
};
const controls = document.getElementById('controls');
for (const dim in DIMS){
  const grp = document.createElement('div'); grp.className = 'fgroup';
  const title = document.createElement('div'); title.className = 'fg-title';
  title.innerHTML = GROUP_TITLES[dim] +
    ` <button class="allbtn" data-dim="${dim}">all 全选</button>`;
  grp.appendChild(title);
  for (const val of DIMS[dim]){
    const lab = document.createElement('label');
    const cb = document.createElement('input');
    cb.type = 'checkbox'; cb.checked = true;
    cb.dataset.dim = dim; cb.dataset.val = val;
    lab.appendChild(cb);
    const meta = GROUP_META[dim];
    if (meta.sw){ const sw = document.createElement('span'); sw.className = 'sw';
      sw.style.background = meta.sw(val); lab.appendChild(sw); }
    else if (meta.mg){ const mg = document.createElement('span'); mg.className = 'mg';
      mg.textContent = meta.mg(val); lab.appendChild(mg); }
    const sp = document.createElement('span'); sp.textContent = meta.lbl[val] || val;
    lab.appendChild(sp);
    cb.addEventListener('change', () => toggleVal(dim, val, cb.checked));
    grp.appendChild(lab);
  }
  controls.appendChild(grp);
}

function syncUI(){
  document.querySelectorAll('#controls input[type=checkbox]').forEach(cb => {
    cb.checked = sel[cb.dataset.dim].has(cb.dataset.val);
  });
  document.querySelectorAll('.legend').forEach(g => {
    const dim = g.dataset.dim;
    g.classList.toggle('off', !DIMS[dim].some(v => sel[dim].has(v)));
  });
  document.querySelectorAll('.lg-row').forEach(row => {
    row.classList.toggle('off', !sel[row.dataset.dim].has(row.dataset.val));
  });
}
function toggleVal(dim, val, on){
  if (on) sel[dim].add(val); else sel[dim].delete(val);
  syncUI(); apply();
}
document.querySelectorAll('.allbtn').forEach(b => b.addEventListener('click', () => {
  const dim = b.dataset.dim; sel[dim] = new Set(DIMS[dim]); syncUI(); apply();
}));
document.querySelectorAll('.lg-row').forEach(row =>
  row.addEventListener('click', ev => {
    ev.stopPropagation();
    const dim = row.dataset.dim, val = row.dataset.val;
    toggleVal(dim, val, !sel[dim].has(val));
  }));
document.getElementById('reset').addEventListener('click', () => {
  for (const d in DIMS) sel[d] = new Set(DIMS[d]);
  searchStr = ''; document.getElementById('search').value = '';
  mode = 'reflow';
  const r = document.querySelector('input[name=mode][value=reflow]'); if (r) r.checked = true;
  selectStudy(null); syncUI(); apply();
});
let debounce = null;
document.getElementById('search').addEventListener('input', e => {
  clearTimeout(debounce);
  debounce = setTimeout(() => { searchStr = e.target.value.trim().toLowerCase(); apply(); }, 120);
});
document.querySelectorAll('input[name=mode]').forEach(r =>
  r.addEventListener('change', () => { mode = r.value; apply(); }));

syncUI();
apply();
selectStudy(STUDIES.length ? STUDIES[0].id : null, false);
</script>
</body>
</html>
"""


def export_interactive_html(out_html, studies, geom, counts_summary):
    """Write a single-file interactive SVG+HTML edition of the timeline.

    Reuses the Pass-2 measured card geometries for a pixel-identical initial
    view, then re-runs the same packing solver in JavaScript whenever the
    reader toggles legend entries / filters, and renders one standalone study
    card per publication (6 legend dimensions + participant count + the four
    structured metrics) below the timeline.
    """
    import html as html_lib
    meta = counts_summary.get("metadata", {})
    meta_out = {
        "title": str(meta.get("title", "Living Timeline of Communication Neuroprostheses")),
        "total_unique": counts_summary.get("total_unique", 0),
        "home_summary": str(counts_summary.get("home_summary", "")),
    }
    studies_json = json.dumps(studies, ensure_ascii=False).replace("</", "<\\/")
    doc = (_HTML_TEMPLATE
           .replace("__CHROME__", _build_chrome_svg(geom))
           .replace("__STUDIES__", studies_json)
           .replace("__GEOM__", json.dumps(geom, ensure_ascii=False).replace("</", "<\\/"))
           .replace("__META__", json.dumps(meta_out, ensure_ascii=False).replace("</", "<\\/"))
           .replace("__TITLE__", html_lib.escape(meta_out["title"]))
           .replace("__VW__", f'{geom["sx1"] + 8:.0f}')
           .replace("__VH__", f'{geom["sy1"] + 8:.0f}'))
    with open(out_html, "w", encoding="utf-8", newline="\n") as f:
        f.write(doc)


def main():
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
