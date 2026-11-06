#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Weekly literature patrol and preprint-graduation checker for
`communication-neuroprostheses-timeline`.

1. Audits every `preprint` / `arXiv` / `10.1101/*` entry in `studies.yaml`
   via the Crossref & Europe PMC APIs to detect whether a peer-reviewed
   journal version has been published (`is-preprint-of` relation).
2. Queries Europe PMC (PubMed + bioRxiv + medRxiv) and arXiv over the past
   N days (default: 7) for new speech, typing, and handwriting
   neuroprosthesis studies and publications from tracked clinical BCI labs.
3. Deduplicates against existing entries in `studies.yaml` and writes a
   ready-to-review Markdown report with pre-filled YAML candidate blocks.

Usage:
    python weekly_patrol.py --days 7 --output weekly_candidates.md
"""
import argparse
import datetime as dt
import json
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import yaml

USER_AGENT = "CommunicationNeuroprosthesesTimeline/1.0 (https://github.com/communication-neuroprostheses-timeline)"

EUROPE_PMC_QUERY = (
    '("speech neuroprosthesis" OR "brain-to-text" OR "brain-to-voice" '
    'OR "handwriting neuroprosthesis" OR "typing neuroprosthesis" '
    'OR ("speech decoding" AND ("intracortical" OR "ECoG" OR "Utah array" OR "electrocorticography")) '
    'OR ("inner speech" AND ("intracortical" OR "Utah array" OR "microelectrode")))'
)

ARXIV_QUERY = (
    'all:"brain-to-text" OR all:"speech neuroprosthesis" OR all:"brain-to-voice" '
    'OR (all:"speech decoding" AND (all:"intracortical" OR all:"Utah array" OR all:"ECoG"))'
)


def _http_get_json(url, timeout=15):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _http_get_text(url, timeout=15):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def check_preprint_graduations(studies):
    """Check if any preprint in `studies.yaml` now has a linked journal DOI in Crossref."""
    graduations = []
    for st in studies:
        doi = str(st.get("doi", "")).strip()
        avail = st.get("data_availability", "")
        if avail != "preprint" and not doi.startswith("10.1101/"):
            continue
        if not doi or doi.startswith("10.48550/arXiv"):
            continue
        url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"
        try:
            data = _http_get_json(url)
            msg = data.get("message", {})
            rel = msg.get("relation", {})
            preprint_of = rel.get("is-preprint-of", [])
            if preprint_of:
                journal_dois = [x.get("id") for x in preprint_of if x.get("id")]
                graduations.append({
                    "id": st["id"],
                    "citation": st["citation"],
                    "preprint_doi": doi,
                    "journal_dois": journal_dois,
                })
        except Exception as exc:
            graduations.append({
                "id": st["id"],
                "citation": st["citation"],
                "preprint_doi": doi,
                "error": str(exc),
            })
    return graduations


def query_europe_pmc(days=7, max_results=25):
    """Search Europe PMC (PubMed + bioRxiv + medRxiv) for recent papers."""
    today = dt.date.today()
    since = today - dt.timedelta(days=days)
    date_filter = f"FIRST_PDATE:[{since.isoformat()} TO {today.isoformat()}]"
    full_q = f"({EUROPE_PMC_QUERY}) AND {date_filter}"
    params = urllib.parse.urlencode({
        "query": full_q,
        "format": "json",
        "pageSize": str(max_results),
        "sort": "P_PDATE_D desc",
    })
    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?{params}"
    try:
        data = _http_get_json(url)
    except Exception as exc:
        return [], str(exc)

    results = []
    for item in data.get("resultList", {}).get("result", []):
        results.append({
            "source": f"EuropePMC ({item.get('source', 'unknown')})",
            "title": item.get("title", "").strip(),
            "authorString": item.get("authorString", "").strip(),
            "journalTitle": item.get("journalTitle") or item.get("bookOrReportDetails", {}).get("publisher", "preprint"),
            "pubDate": item.get("firstPublicationDate", ""),
            "doi": item.get("doi", ""),
        })
    return results, None


def query_arxiv(days=7, max_results=20):
    """Search arXiv API for recent brain-to-text / speech neuroprosthesis preprints."""
    params = urllib.parse.urlencode({
        "search_query": ARXIV_QUERY,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": str(max_results),
    })
    url = f"https://export.arxiv.org/api/query?{params}"
    try:
        xml_text = _http_get_text(url)
    except Exception as exc:
        return [], str(exc)

    cutoff = dt.date.today() - dt.timedelta(days=days)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_text)
    results = []
    for entry in root.findall("atom:entry", ns):
        pub_raw = (entry.findtext("atom:published", default="", namespaces=ns) or "")[:10]
        try:
            pub_date = dt.date.fromisoformat(pub_raw)
        except ValueError:
            continue
        if pub_date < cutoff:
            continue
        title = " ".join((entry.findtext("atom:title", default="", namespaces=ns) or "").split())
        authors = [a.findtext("atom:name", default="", namespaces=ns)
                   for a in entry.findall("atom:author", ns)]
        aid = (entry.findtext("atom:id", default="", namespaces=ns) or "").split("/abs/")[-1]
        results.append({
            "source": "arXiv",
            "title": title,
            "authorString": ", ".join(authors[:4]) + (" et al." if len(authors) > 4 else ""),
            "journalTitle": "arXiv",
            "pubDate": pub_raw,
            "doi": f"10.48550/arXiv.{aid.split('v')[0]}" if aid else "",
        })
    return results, None


def format_report(studies_doc, days, graduations, candidates, errors):
    today = dt.date.today().isoformat()
    studies = studies_doc.get("studies", [])
    lines = [
        f"# Weekly Living Timeline Patrol Report ({today})",
        "",
        f"- **Repository**: `communication-neuroprostheses-timeline`",
        f"- **Lookback window**: last `{days}` days",
        f"- **Current timeline cards**: `{len(studies)}` studies",
        "",
        "## 1. Preprint-to-Journal Graduation Check",
        "",
    ]
    grad_hits = [g for g in graduations if g.get("journal_dois")]
    if not grad_hits:
        lines.append(
            "No tracked bioRxiv/medRxiv preprints reported a newly linked journal DOI this week."
        )
    else:
        for g in grad_hits:
            j_str = ", ".join(f"`{d}`" for d in g["journal_dois"])
            lines.append(
                f"- **`{g['id']}` ({g['citation']})**: preprint `{g['preprint_doi']}` "
                f"is now linked to published journal DOI {j_str}. "
                f"Update `data_availability` (`open` / `on_request` / `closed`) and `doi` in `studies.yaml`."
            )
    lines.append("")
    lines.append("## 2. Newly Discovered Candidate Papers")
    lines.append("")
    if errors:
        for src, err in errors:
            lines.append(f"> Note: `{src}` query returned warning: `{err}`")
        lines.append("")

    if not candidates:
        lines.append("No new candidate papers outside `studies.yaml` were found in this window.")
    else:
        for idx, c in enumerate(candidates, 1):
            first_author = (c["authorString"].split(",")[0].split()[0]
                            if c["authorString"] else "Author")
            yr = (c["pubDate"] or today)[:4]
            slug = f"{first_author.lower()}{yr}"
            lines.extend([
                f"### {idx}. {c['title']}",
                f"- **Authors**: {c['authorString']}",
                f"- **Venue / Source**: {c['journalTitle']} ({c['source']})",
                f"- **Date**: `{c['pubDate']}`",
                f"- **DOI**: `{c['doi'] or 'n/a'}`",
                "",
                "Draft `studies.yaml` snippet (verify inclusion criteria & metrics before merging):",
                "```yaml",
                f"  - id: {slug}",
                f"    citation: \"{first_author} et al., {yr}\"",
                f"    date: \"{c['pubDate'] or today}\"",
                "    lane: speech  # speech | typing | handwriting",
                "    electrode: utah  # utah | ecog | thin_film | microwire",
                "    data_availability: preprint  # open | on_request | closed | preprint | reanalysis",
                "    decoding_mode: offline  # online | offline",
                "    task_instruction: intended  # inner | intended | both",
                "    decoded_output: text  # voice | text | both",
                "    descriptor: \"<short system descriptor>\"",
                "    metrics:",
                "      vocab_size: \"vocab <y>\"",
                "      sentences: \"<N> sentences (<M> sessions)\"",
                "      wer: \"WER <x>%\"",
                "      speed: \"<z> wpm\"  # or \"n/r\"",
                "    participants: [\"T12△\"]",
                f"    doi: \"{c['doi']}\"",
                "```",
                "",
            ])
    return "\n".join(lines) + "\n"


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--yaml", default=os.path.join(here, "studies.yaml"),
                    help="Path to studies.yaml")
    ap.add_argument("--days", type=int, default=7,
                    help="Lookback window in days (default: 7)")
    ap.add_argument("--output", default="",
                    help="Optional path to write Markdown report")
    ap.add_argument("--offline-only", action="store_true",
                    help="Skip live network calls and only validate local studies.yaml")
    args = ap.parse_args()

    with open(args.yaml, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    studies = doc.get("studies", [])
    known_dois = {str(s.get("doi", "")).lower().strip() for s in studies if s.get("doi")}

    graduations, candidates, errors = [], [], []
    if not args.offline_only:
        graduations = check_preprint_graduations(studies)
        epmc_res, epmc_err = query_europe_pmc(days=args.days)
        if epmc_err:
            errors.append(("EuropePMC", epmc_err))
        arxiv_res, arxiv_err = query_arxiv(days=args.days)
        if arxiv_err:
            errors.append(("arXiv", arxiv_err))

        seen = set(known_dois)
        for item in epmc_res + arxiv_res:
            d = item.get("doi", "").lower().strip()
            key = d or item.get("title", "").lower()
            if key and key not in seen:
                seen.add(key)
                candidates.append(item)

    report_md = format_report(doc, args.days, graduations, candidates, errors)
    if args.output:
        with open(args.output, "w", encoding="utf-8", newline="\n") as f:
            f.write(report_md)
        print(f"[ok] Wrote weekly patrol report to {args.output} "
              f"({len(candidates)} new candidates, {len(graduations)} preprints checked)")
    else:
        print(report_md)


if __name__ == "__main__":
    main()
