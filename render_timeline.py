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
    print(f"     Cohort: {summary['total_unique']} unique participants "
          f"({summary['lane_totals']['speech']} speech, "
          f"{summary['lane_totals']['typing']} typing, "
          f"{summary['lane_totals']['handwriting']} handwriting)")
    print("     Overlap check: ALL CLEAN")


if __name__ == "__main__":
    main()
