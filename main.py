#!/usr/bin/env python3
"""
Shaw Strengths Matrix™ PDF Generator - Cloud Run Service

This is a stateless HTTP service that generates PDF reports from Excel assessment data.
Designed for deployment on Google Cloud Run.

Endpoints:
  POST /generate-pdf-base64
    Body: { "excel_base64": "...", "filename": "..." }
    Returns: { "success": true, "pdf_base64": "...", "filename": "..." }
  
  POST /generate-team-pdf
    Body: { "company_name": "...", "team_name": "...", "num_members": N, 
            "date_str": "YYYY-MM-DD", "individual_results": [...] }
    Returns: { "success": true, "pdf_base64": "...", "filename": "..." }
    
  GET /health
    Returns: { "status": "healthy" }

Team Report Algorithm:
  Step 1: Calculate average ranks across all team members
  Step 2: Sort traits by average rank
  Step 2.1: Convert ranked averages to 1-12 rankings
  Step 2.2: Team Tie-Breaker using Median Rank
    - When two or more traits have the same mean rank, use median of individual ranks
    - Lower median = stronger placement (ranks higher)
  Step 2.3: Average Rankings for remaining ties
    - If traits still tie after median comparison, assign average of nominal positions
    - Example: 3 traits tied for 3rd → ranks (3+4+5)/3 = 4.0 for all three
  Step 3: Calculate distribution data (percentage in each category)
  Step 4: Generate team PDF with distribution chart
"""

import os
import io
import base64
import tempfile
import itertools
import re
from datetime import datetime
from statistics import mean
from typing import Optional, Tuple, Dict, List, Any

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer,
    Image,
    PageBreak,
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Group
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

# ---------------------------
# CONFIG
# ---------------------------
# Logo path - should be in the same directory as this script in Cloud Run
LOGO_PATH = os.path.join(os.path.dirname(__file__), "logo.png")

TRAITS = [
    "Courage",
    "Practicality",
    "Curiosity",
    "Prudence",
    "Confidence",
    "Discernment",
    "Fairness",
    "Tenacity",
    "Foresight",
    "Discipline",
    "Objectivity",
    "Empathy",
]

DESCRIPTIONS = {
    "Fairness": "The <b>thinking</b> ability to apply impartiality and equitable treatment, guided by <b>anticipation</b> of imminent outcomes.",
    "Empathy": "The <b>feeling</b> ability to understand others' perspectives and experiences, guided by <b>reflection</b> on recent cues.",
    "Discernment": "The <b>sensory</b> ability to identify and distinguish important details, guided by <b>reflection</b> on recent cues.",
    "Objectivity": "The <b>thinking</b> ability to process information rationally and without bias to achieve fact-based knowledge, guided by <b>awareness</b> of the current situation.",
    "Tenacity": "The <b>feeling</b> ability to apply persistent and determined effort to overcome obstacles, guided by <b>awareness</b> of the current situation.",
    "Courage": "The <b>intuitive</b> ability to take purposeful action despite fear or uncertainty. guided by <b>anticipation</b> of imminent outcomes.",
    "Confidence": "The <b>intuitive</b> ability to convey self-assurance and assertiveness that inspires respect and persuades others, guided by <b>awareness</b> of the current situation.",
    "Prudence": "The <b>feeling</b> ability to apply careful and discreet judgment, guided by <b>anticipation</b> of imminent outcomes.",
    "Foresight": "The <b>intuitive</b> ability to project future trends and opportunities, guided by <b>reflection</b> on recent cues.",
    "Practicality": "The <b>sensory</b> ability to take calm, appropriate and effective action to address real-world needs, guided by <b>awareness</b> of the current situation.",
    "Curiosity": "The <b>thinking</b> ability to explore and learn new information, guided by <b>reflection</b> on recent cues.",
    "Discipline": "The <b>sensory</b> ability to apply self-regulation and consistency, guided by <b>anticipation</b> of imminent outcomes.",
}

ONET_STYLES = {
    "Confidence": (
        "Self-<br/>Confidence",
        "A tendency to believe in one's work-related capabilities and ability to control one's work-related outcomes."
    ),
    "Courage": (
        "Initiative",
        "A tendency to be proactive and take on extra responsibilities and tasks that may fall outside of one's required work role."
    ),
    "Curiosity": (
        "Intellectual Curiosity",
        "A tendency to seek out and acquire new work-related knowledge and obtain a deep understanding of work-related subjects."
    ),
    "Discernment": (
        "Attention to Detail",
        "A tendency to be detailed oriented, organised, and thorough in completing work."
    ),
    "Discipline": (
        "Dependability",
        "A tendency to be reliable, responsible, and consistently meet work-related obligations."
    ),
    "Empathy": (
        "Empathy",
        "A tendency to show concern for others and be sensitive to others' needs and feelings at work."
    ),
    "Fairness": (
        "Cooperation",
        "A tendency to be pleasant, helpful, and willing to assist others at work."
    ),
    "Foresight": (
        "Innovative",
        "A tendency to be inventive, imaginative, and adopt new perspectives on ways to accomplish work."
    ),
    "Objectivity": (
        "Adaptability",
        "A tendency to be open to and comfortable with change, new experiences, or ideas at work."
    ),
    "Practicality": (
        "Self-Control",
        "A tendency to remain calm, composed, and manage emotions effectively in response to criticism or difficult situations at work."
    ),
    "Prudence": (
        "Cautiousness",
        "A tendency to be careful, deliberate, and risk-avoidant when making work-related decisions or doing work."
    ),
    "Tenacity": (
        "Perseverance",
        "A tendency to exhibit determination and resolve to perform or complete tasks in the face of difficult circumstances or obstacles at work."
    ),
}

ONET_ACTIVITIES = {
    "Confidence":
        "Selling or Influencing Others<br/>Guiding, Directing, and Motivating Subordinates<br/>Communicating with People Outside the Organisation<br/>Performing for or Working Directly with the Public",
    "Courage":
        "Coordinating the Work and Activities of Others<br/>Making Decisions and Solving Problems",
    "Curiosity":
        "Getting Information<br/>Updating and Using Relevant Knowledge<br/>Training and Teaching Others<br/>Interpreting the Meaning of Information for Others",
    "Discernment":
        "Inspecting Equipment, Structures, or Materials<br/>Drafting, Laying Out, and Specifying Technical Devices, Parts, and Equipment<br/>Identifying Objects, Actions, and Events<br/>Evaluating Information to Determine Compliance with Standards",
    "Discipline":
        "Scheduling Work and Activities<br/>Performing Administrative Activities<br/>Monitoring Processes, Materials, or Surroundings<br/>Controlling Machines and Processes",
    "Empathy":
        "Assisting and Caring for Others<br/>Coaching and Developing Others",
    "Fairness":
        "Establishing and Maintaining Interpersonal Relationships<br/>Developing and Building Teams<br/>Communicating with Supervisors, Peers, or Subordinates<br/>Staffing Organisational Units",
    "Foresight":
        "Thinking Creatively<br/>Developing Objectives and Strategies<br/>Organising, Planning, and Prioritising Work<br/>Estimating the Quantifiable Characteristics of Products, Events, or Information",
    "Objectivity":
        "Providing Consultation and Advice to Others<br/>Working with Computers<br/>Analysing Data or Information<br/>Processing Information",
    "Practicality":
        "Operating Vehicles, Mechanised Devices, or Equipment<br/>Handling and Moving Objects",
    "Prudence":
        "Monitoring and Controlling Resources<br/>Documenting/Recording Information<br/>Judging the Qualities of Objects, Services, or People",
    "Tenacity":
        "Resolving Conflicts and Negotiating with Others<br/>Repairing and Maintaining Mechanical Equipment<br/>Repairing and Maintaining Electronic Equipment<br/>Performing General Physical Activities",
}


# ---------------------------
# Sheet Resolution
# ---------------------------
def _resolve_sheets(xls_path: str) -> Tuple[str, Optional[str]]:
    """
    Returns (instr_sheet_name, survey_sheet_name)
      - If 'Instructions' exists (case-insensitive), use it; else use 1st sheet.
      - If 'Survey' exists (case-insensitive), use it; else use 2nd sheet; if only one sheet, reuse the 1st.
    """
    xls = pd.ExcelFile(xls_path)
    names = xls.sheet_names
    lower = {s.strip().lower(): s for s in names}

    instr = lower.get("instructions", names[0])
    if "survey" in lower:
        survey = lower["survey"]
    else:
        survey = names[1] if len(names) >= 2 else names[0]
    return instr, survey


# ---------------------------
# Identity Extraction
# ---------------------------
def extract_identity_flexible(xls_path: str) -> Tuple[str, str, str]:
    """
    Extracts Name and Date from the Instructions sheet.
    Returns (first, last, date_str in YYYY-MM-DD).
    """
    instr_sheet, _ = _resolve_sheets(xls_path)

    df = pd.read_excel(xls_path, sheet_name=instr_sheet, header=None, dtype=str)
    cells = [str(x) for x in df.values.ravel() if pd.notna(x)]
    blob = "\n".join(cells)

    first, last = "First", "Last"
    date_str = datetime.today().strftime("%Y-%m-%d")

    STOPWORDS = {
        "if", "your", "please", "enter", "write", "type", "here", "personality",
        "assessment", "instructions", "sheet", "survey", "participant", "name",
        "first", "surname", "last",
    }

    def clean_frag(s: str) -> str:
        s = re.sub(r"[\s_.\-–—]+$", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def tokens_alpha(s: str) -> List[str]:
        return re.findall(r"[A-Za-z][A-Za-z'\.\-]*", s)

    def is_name_like(s: str) -> bool:
        toks = tokens_alpha(s)
        if not (1 <= len(toks) <= 4):
            return False
        if {t.lower() for t in toks} & STOPWORDS:
            return False
        return any(
            (t[:1].isupper() and (t[1:].islower() or t[1:] in {"'", "-", ".", ""}))
            or (t.isupper() and len(t) <= 3)
            for t in toks
        )

    # 1) Column-style: First Name / Surname
    # Excel structure: ['First Name', value] in row 0, ['Surname', value] in row 1
    first_val = None
    last_val = None
    
    # First, try direct column-based lookup (most reliable for our format)
    # Excel structure: ['First Name', value] in row 0, ['Surname', value] in row 1
    for r in range(min(10, len(df))):  # Only check first 10 rows
        row = [str(v) if pd.notna(v) else "" for v in df.iloc[r, :].tolist()]
        
        # Check if this row has "First Name" or "Surname" in first column
        if len(row) >= 2:
            label = row[0].strip()
            value = row[1].strip() if len(row) > 1 else ""
            
            # Check for First Name - accept any non-empty value that's not a placeholder
            if first_val is None and re.search(r"(?i)^\s*(first\s*name|firstname)\s*$", label):
                cleaned = clean_frag(value)
                # Accept if it's not empty and not a known placeholder
                if cleaned and cleaned.lower() not in ["first", "participant", "name", ""]:
                    first_val = cleaned
            
            # Check for Surname - accept any non-empty value that's not a placeholder
            if last_val is None and re.search(r"(?i)^\s*(surname|last\s*name|lastname)\s*$", label):
                cleaned = clean_frag(value)
                # Accept if it's not empty and not a known placeholder
                if cleaned and cleaned.lower() not in ["last", "name", "surname", ""]:
                    last_val = cleaned
    
    # Fallback: More flexible parsing if direct lookup didn't work
    if not (first_val and last_val):
        for r in range(min(150, len(df))):
            row = [str(v) if pd.notna(v) else "" for v in df.iloc[r, :].tolist()]
            for c in range(len(row)):
                cell = row[c].strip()

                if first_val is None:
                    m = re.match(
                        r"(?i)^\s*(first\s*name|firstname|first)\b\s*[:\-]?\s*(.+)$", cell
                    )
                    if m:
                        cand = clean_frag(m.group(2))
                        if cand and cand.lower() not in ["first", "participant"] and is_name_like(cand):
                            first_val = cand
                    elif c + 1 < len(row) and re.search(
                        r"(?i)\b(first\s*name|firstname|first)\b", cell
                    ):
                        cand = clean_frag(row[c + 1])
                        if cand and cand.lower() not in ["first", "participant"] and is_name_like(cand):
                            first_val = cand

                if last_val is None:
                    m = re.match(
                        r"(?i)^\s*(surname|last\s*name|lastname|last)\b\s*[:\-]?\s*(.+)$",
                        cell,
                    )
                    if m:
                        cand = clean_frag(m.group(2))
                        if cand and cand.lower() not in ["last", "name"] and is_name_like(cand):
                            last_val = cand
                    elif c + 1 < len(row) and re.search(
                        r"(?i)\b(surname|last\s*name|lastname|last)\b", cell
                    ):
                        cand = clean_frag(row[c + 1])
                        if cand and cand.lower() not in ["last", "name"] and is_name_like(cand):
                            last_val = cand

            if first_val and last_val:
                break

    # 2) Legacy single-cell: "Name: Jason Loosle________"
    if not (first_val and last_val):
        for cell in cells:
            m = re.match(r"(?i)^\s*name\s*[:\-]?\s*(.+?)\s*[_\-\s]*$", cell.strip())
            if m:
                cand = clean_frag(m.group(1))
                if is_name_like(cand):
                    toks = tokens_alpha(cand)
                    if toks:
                        first_val = first_val or toks[0]
                        if len(toks) > 1:
                            last_val = last_val or toks[-1]
                    break

    if first_val:
        # Use the full cleaned value, not just the first token
        first = clean_frag(first_val)
        # If it's still empty or just whitespace, keep default
        if not first or first.lower() in ["first", "participant"]:
            first = "First"
    if last_val:
        # Use the full cleaned value, not just the last token
        last = clean_frag(last_val)
        # If it's still empty or just whitespace, keep default
        if not last or last.lower() in ["last", "name"]:
            last = "Last"

    # --- DATE parsing ---
    def parse_date_any(raw: str) -> Optional[str]:
        raw = re.sub(r"[ _]", "", raw)
        for fmt in ("%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d", "%m/%d/%y", "%d/%m/%y"):
            try:
                dt = datetime.strptime(raw, fmt)
                if "%y" in fmt and dt.year < 1950:
                    dt = dt.replace(year=dt.year + 2000)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
        return None

    for cell in cells:
        m = re.match(
            r"(?i)^\s*date[^\n:]*[:\-]?\s*([_\s\d/]+?)\s*[_\s]*$", cell.strip()
        )
        if m:
            parsed = parse_date_any(m.group(1))
            if parsed:
                date_str = parsed
                break
    else:
        m = re.search(r"(\d{1,4}\s*/\s*\d{1,2}\s*/\s*\d{2,4})", blob, flags=re.I)
        if m:
            parsed = parse_date_any(m.group(1))
            if parsed:
                date_str = parsed

    return first, last, date_str


# ---------------------------
# Survey Parsing
# ---------------------------
def parse_survey_flexible(xls_path: str, traits_set=None) -> Dict[str, Dict[str, int]]:
    """
    Parse the Survey sheet and extract win/loss results for each trait pair.
    """
    if traits_set is None:
        traits_set = set(TRAITS)

    _, survey_sheet = _resolve_sheets(xls_path)
    raw = pd.read_excel(xls_path, sheet_name=survey_sheet, header=None)

    # Choose a header row
    hdr_idx = None
    for i in range(min(120, len(raw))):
        vals = raw.iloc[i].astype(str).str.strip().str.lower().tolist()
        if any("test" in v for v in vals) or (
            any("construct a" in v for v in vals)
            and any("construct b" in v for v in vals)
        ):
            hdr_idx = i
            break
    if hdr_idx is None:
        for i in range(min(10, len(raw))):
            if raw.iloc[i].notna().any():
                hdr_idx = i
                break

    df = pd.read_excel(xls_path, sheet_name=survey_sheet, header=hdr_idx)
    cols = {str(c).strip().lower(): c for c in df.columns}

    def col(*keys):
        for name, orig in cols.items():
            if any(k in name for k in keys):
                return orig
        return None

    col_q = col("paired", "pair", "question", "#")
    col_test = col("test")
    col_a = col("construct a", "a (left)", "left")
    col_b = col("construct b", "b (right)", "right")

    ans_cols = [orig for name, orig in cols.items() if "response" in name or "answer" in name]

    if col_q is None:
        df["_pair_idx__"] = (df.index // 2) + 1
        col_q = "_pair_idx__"
    else:
        df[col_q] = df[col_q].ffill()

    def parse_test(text: str) -> Tuple[Optional[str], Optional[str]]:
        m = re.search(
            r"^\s*([A-Za-z \-']+?)\s*vs\.?\s*([A-Za-z \-']+?)\s*$",
            str(text or ""),
            re.IGNORECASE,
        )
        if not m:
            return None, None
        return m.group(1).strip().title(), m.group(2).strip().title()

    def row_has_X_anywhere(row) -> bool:
        for v in row.values:
            s = str(v).strip().upper()
            if s == "X":
                return True
        return False

    results = {t: {u: 0 for u in TRAITS if u != t} for t in TRAITS}
    wins = 0

    for _, g in df.groupby(col_q, dropna=True):
        g = g.copy()

        left_idx = right_idx = None
        if col_a is not None:
            na = g[g[col_a].notna()]
            if not na.empty:
                left_idx = int(na.index[0])
        if col_b is not None:
            nb = g[g[col_b].notna()]
            if not nb.empty:
                right_idx = int(nb.index[0])

        if left_idx is None or right_idx is None:
            idxs = list(g.index)
            if len(idxs) >= 2:
                if left_idx is None:
                    left_idx = int(idxs[0])
                if right_idx is None:
                    right_idx = int(idxs[1])
            else:
                continue

        L = R = None
        if col_test is not None:
            tt = None
            for _, r in g.iterrows():
                v = r.get(col_test, "")
                if isinstance(v, str) and "vs" in v.lower():
                    tt = v
                    break
            if tt:
                L, R = parse_test(tt)

        if (L is None or R is None) and (col_a is not None and col_b is not None):
            L = (
                str(df.loc[left_idx, col_a])
                if pd.notna(df.loc[left_idx, col_a])
                else None
            )
            R = (
                str(df.loc[right_idx, col_b])
                if pd.notna(df.loc[right_idx, col_b])
                else None
            )
            L = L.strip().title() if L else None
            R = R.strip().title() if R else None

        if not L or not R or L == R or L not in traits_set or R not in traits_set:
            continue

        winner = None

        if ans_cols:
            mask_any = None
            for ac in ans_cols:
                m = g[ac].astype(str).str.strip().str.upper().eq("X")
                mask_any = m if mask_any is None else (mask_any | m)
            if mask_any is not None and mask_any.any():
                x_idx = int(mask_any[mask_any].index[0])
                if x_idx == left_idx:
                    winner = L
                elif x_idx == right_idx:
                    winner = R

        if winner is None:
            left_row = df.loc[left_idx]
            right_row = df.loc[right_idx]
            left_has = row_has_X_anywhere(left_row)
            right_has = row_has_X_anywhere(right_row)
            if left_has ^ right_has:
                winner = L if left_has else R

        if winner is None:
            continue

        loser = R if winner == L else L
        results[winner][loser] = 1
        wins += 1

    return results


# ---------------------------
# Tie-breaker (5 steps)
# ---------------------------
class TieBreaker:
    def __init__(self, traits, results):
        self.traits = list(traits)
        self.results = results
        self.scores = {t: sum(results[t].values()) for t in traits}

    def rank_with_average_ranks(self) -> Tuple[List[str], Dict[str, float]]:
        base = self._primary_order()
        groups = self._resolve_equal_score_groups(base)
        return self._assign_average_ranks(groups)

    def _primary_order(self) -> List[str]:
        return sorted(self.traits, key=lambda t: (-self.scores[t], t))

    def _resolve_equal_score_groups(self, base_order) -> List:
        out = []
        i = 0
        while i < len(base_order):
            run = [base_order[i]]
            while (
                i + len(run) < len(base_order)
                and self.scores[base_order[i]] == self.scores[base_order[i + len(run)]]
            ):
                run.append(base_order[i + len(run)])

            if len(run) == 1:
                out.append(run[0])
            elif len(run) == 2:
                a, b = run
                out.extend(self._head_to_head_pair(a, b))
            else:
                out.extend(self._order_run_with_minileague(run))
            i += len(run)
        return out

    def _head_to_head_pair(self, a, b) -> List[str]:
        return [a, b] if self.results[a].get(b, 0) == 1 else [b, a]

    def _order_run_with_minileague(self, run) -> List:
        subs = {t: 0 for t in run}
        for a, b in itertools.permutations(run, 2):
            subs[a] += self.results[a].get(b, 0)

        sorted_by_sub = sorted(run, key=lambda t: (-subs[t], t))
        resolved = []
        i = 0
        while i < len(sorted_by_sub):
            block = [sorted_by_sub[i]]
            while (
                i + len(block) < len(sorted_by_sub)
                and subs[sorted_by_sub[i]] == subs[sorted_by_sub[i + len(block)]]
            ):
                block.append(sorted_by_sub[i + len(block)])

            if len(block) == 1:
                resolved.append(block[0])
            elif len(block) == 2:
                a, b = block
                resolved.extend(self._head_to_head_pair(a, b))
            else:
                resolved.extend(self._mini_league_then_sos_with_extra(block))
            i += len(block)
        return resolved

    def _mini_league_then_sos_with_extra(self, block) -> List:
        wins = {t: 0 for t in block}
        for a, b in itertools.permutations(block, 2):
            wins[a] += self.results[a].get(b, 0)

        ordered = sorted(block, key=lambda t: (-wins[t], t))
        out = []
        i = 0
        while i < len(ordered):
            band = [ordered[i]]
            while (
                i + len(band) < len(ordered)
                and wins[ordered[i]] == wins[ordered[i + len(band)]]
            ):
                band.append(ordered[i + len(band)])

            if len(band) == 1:
                out.append(band[0])
            elif len(band) == 2:
                a, b = band
                out.extend(self._head_to_head_pair(a, b))
            else:
                out.extend(self._sos_then_extra_minileague_on_ties(band))
            i += len(band)
        return out

    def _sos_then_extra_minileague_on_ties(self, band) -> List:
        sos = {t: 0 for t in band}
        for t in band:
            sos[t] = sum(
                self.scores[o]
                for o in self.results[t]
                if self.results[t].get(o, 0) == 1
            )

        ordered = sorted(band, key=lambda t: (-sos[t], t))
        out = []
        i = 0
        while i < len(ordered):
            sub = [ordered[i]]
            while (
                i + len(sub) < len(ordered)
                and sos[ordered[i]] == sos[ordered[i + len(sub)]]
            ):
                sub.append(ordered[i + len(sub)])

            if len(sub) == 1:
                out.append(sub[0])
            elif len(sub) == 2:
                a, b = sub
                out.extend(self._head_to_head_pair(a, b))
            else:
                out.extend(self._final_minileague_or_tie(sub))
            i += len(sub)
        return out

    def _final_minileague_or_tie(self, subband) -> List:
        wins = {t: 0 for t in subband}
        for a, b in itertools.permutations(subband, 2):
            wins[a] += self.results[a].get(b, 0)

        ordered = sorted(subband, key=lambda t: (-wins[t], t))
        out = []
        i = 0
        while i < len(ordered):
            band = [ordered[i]]
            while (
                i + len(band) < len(ordered)
                and wins[ordered[i]] == wins[ordered[i + len(band)]]
            ):
                band.append(ordered[i + len(band)])

            if len(band) == 1:
                out.append(band[0])
            elif len(band) == 2:
                a, b = band
                out.extend(self._head_to_head_pair(a, b))
            else:
                out.append(sorted(band))
            i += len(band)
        return out

    def _assign_average_ranks(self, groups) -> Tuple[List[str], Dict[str, float]]:
        ordered, ranks, pos = [], {}, 1
        for item in groups:
            if isinstance(item, str):
                ordered.append(item)
                ranks[item] = pos
                pos += 1
            else:
                m = len(item)
                avg = mean(range(pos, pos + m))
                for t in sorted(item):
                    ordered.append(t)
                    ranks[t] = avg
                pos += m
        return ordered, ranks


def category_for_rank_number(rank_num: float) -> str:
    """Map rank to Signature/Supporting/Emerging."""
    if rank_num <= 4:
        return "Signature"
    if rank_num <= 8:
        return "Supporting"
    return "Emerging"


# ---------------------------
# PDF Generation
# ---------------------------
def _parse_date_long(iso_date: str):
    """Parse a date from various incoming formats."""
    try:
        return datetime.strptime(iso_date, "%Y-%m-%d")
    except Exception:
        for fmt in ("%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d", "%m/%d/%y", "%d/%m/%y"):
            try:
                dt = datetime.strptime(iso_date, fmt)
                if "%y" in fmt and dt.year < 1950:
                    dt = dt.replace(year=dt.year + 2000)
                return dt
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(iso_date)
        except Exception:
            return iso_date


def generate_pdf(
    first: str,
    last: str,
    date_str: str,
    ordered_traits: List[str],
    ranks: Dict[str, float],
    output_stream: io.BytesIO,
    logo_path: str = LOGO_PATH,
) -> str:
    """
    Generate PDF and write to output_stream.
    Returns the suggested filename.
    """

    def _fmt_rank(r):
        try:
            r = float(r)
            return str(int(r)) if r.is_integer() else f"{r:.1f}"
        except Exception:
            return str(r)

    def _color_for_category(cat: str):
        return colors.HexColor(
            "#4d93d9" if cat == "Signature" else (
            "#94dcf8" if cat == "Supporting" else
            "#d0d0d0")
        )

    date = _parse_date_long(date_str)

    doc = SimpleDocTemplate(
        output_stream, pagesize=A4, leftMargin=40, rightMargin=40, topMargin=36, bottomMargin=36
    )
    styles = getSampleStyleSheet()

    header_style = ParagraphStyle(
        "SSMHeader", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=22, leading=27, alignment=TA_LEFT, spaceAfter=8
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"], fontName="Helvetica",
        fontSize=11, leading=13, alignment=TA_LEFT
    )
    body_bold_style = ParagraphStyle("CellBold", parent=body_style, fontName="Helvetica-Bold")
    body_right_style = ParagraphStyle("AsideRight", parent=body_style, alignment=TA_RIGHT)
    cell_style = ParagraphStyle("Cell", parent=body_style, fontSize=9, leading=10)
    cell_bold_style = ParagraphStyle("CellBold", parent=cell_style, fontName="Helvetica-Bold")
    cell_center_style = ParagraphStyle("CellCenter", parent=cell_style, alignment=TA_CENTER)
    cell_bold_center_style = ParagraphStyle("CellBoldCenter", parent=cell_bold_style, alignment=TA_CENTER)
    
    table_border = TableStyle([
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
    ])

    story = []
    
    # Page 1 - Title page
    story.append(Table(
        [[Paragraph("Shaw Strengths Matrix™ (SSM™)<br/>Assessment", header_style)]],
        style=table_border
    ))

    story.append(Spacer(1, 6))

    story.append(Table([[Paragraph("", style=body_style)]], style=TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#ba9354"))
    ])))
    
    story.append(Table([
        [Paragraph("Report prepared for:", style=body_bold_style)],
        [Paragraph(f"Name: {first} {last}", style=body_bold_style)],
        [Paragraph(f"Date: {date: %d %B %Y}", style=body_bold_style)],
        ],
        style=table_border,
    ))
    story.append(Spacer(1, 78))

    if os.path.exists(logo_path):
        logo_img = Image(logo_path, width=1.54*inch, height=0.79*inch, kind="proportional")
    else:
        logo_img = Paragraph("", styles["Normal"])

    story.append(Table([[
        logo_img,
        Paragraph("ShawSight Pty Ltd   |   ABN:  38688414557", style=body_style),
    ]]))
    story.append(Spacer(1,6))
    
    legal_notices = Paragraph(
        """
        <i>Shaw Strengths Matrix™ Framework</i> Copyright 2025 by ShawSight Pty Ltd. All rights reserved.<br/>
        <i>Shaw Strengths Matrix™ Assessment</i> Copyright 2025 by ShawSight Pty Ltd. All rights reserved.<br/>
        <i>ShawSight</i> logo is Copyright 2025 by ShawSight Pty Ltd. All rights reserved.<br/>
        No part of this publication may be reproduced in any form or manner without prior written permission from ShawSight Pty Ltd.<br/>
        O*NET is a trademark of the U.S. Department of Labor, Employment and Training Administration.""",
        style=body_style
    )
    story.append(Table([[legal_notices]], style=table_border))

    story.append(PageBreak())

    # Header template for pages 2-6
    def header_template(page_num: int, subtitle: str) -> Table:
        return Table(
            [[
                Paragraph("Shaw Strengths Matrix™ Assessment", style=body_style),
                Paragraph(f"{first} {last} | Page {page_num}", style=body_right_style),
            ], [
                Paragraph(f"Shaw Strengths Matrix™<br/>{subtitle}", style=header_style),
                ""
            ]],
            style=TableStyle([
                ("BOX", (0, 1), (1, 1), 1, colors.black),
                ("SPAN", (0, 1), (1, 1)),
            ]),
        )

    # Page 2 - Overview
    story.append(header_template(2, "Overview"))
    story.append(Spacer(1, 6))

    story.append(Table([[Paragraph(
        """
        <b>About This Report</b><br/>
        <br/>
        This report provides an overview of your <b>SSM™ Assessment character strengths</b> profile and how it may relate to patterns of <b>communication, collaboration, and overall team dynamics</b> at work. It is designed to support <b>self-awareness, reflection, and developmental discussion</b> in team and workshop settings.<br/>
        <br/>
        The <b>SSM™ Assessment</b> measures <b>personality</b> — how you tend to behave — rather than <b>abilities or skills</b> (what you are good at) or <b>interests</b> (what you enjoy doing). It is <b>not</b> intended to provide any clinical diagnosis.<br/>
        <br/>
        While the Assessment is grounded in established <b>personality and work style research</b> and has passed <b>content and face validity testing</b>, it is currently undergoing further psychometric validation. Results should therefore be interpreted as <b>insightful tendencies</b> rather than predictive measures, and are <b>not intended</b> for hiring, promotion, or other HR decision-making.<br/>
        <br/>
        In this report, your <b>SSM™ Assessment</b> profile is also conceptually aligned to <b>O*NET Work Styles</b> and <b>Work Activities</b>.<br/>
        <br/>
        The <b>O*NET Resource Center</b> is a professional workforce research portal providing data, tools, technical documentation, and support. It is widely recognised as a <b>global standard in workplace metrics</b>.
        """,
        style=body_style
    )]], style=table_border))
    story.append(Spacer(1, 6))
    
    story.append(Table([[Paragraph(
        """
        <b>Report Contents</b><br/><br/>
        1) Shaw Strengths Matrix™<br/>
        2) Shaw Strengths Matrix™ Assessment Table<br/>
        3) Shaw Strengths Matrix™ Mapping to O*NET Work Styles<br/>
        4) Shaw Strengths Matrix™ Mapping to O*NET Work Activities
        """,
        style=body_style
    )]], style=table_border))
    story.append(PageBreak())

    # Page 3 - Matrix explanation
    story.append(header_template(3, ""))
    story.append(Spacer(1, 12))
    
    story.append(Table(
        [
            ["Shaw Strengths Matrix™","","","",""],
            ["Character Strengths","","Temporal Preferences","",""],
            ["","", "Past Reflections", "Present Awareness", "Future Anticipations"],
            ["Cognitive\nPreferences", "Intuition", "Foresight", "Confidence", "Courage"],
            ["", "Thinking", "Curiosity", "Objectivity", "Fairness"],
            ["", "Feeling", "Empathy", "Tenacity", "Prudence"],
            ["", "Sensing", "Discernment", "Practicality", "Discipline"]
        ],
        style=TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("FONT", (0, 0), (-1, -1), cell_bold_style.fontName, cell_bold_style.fontSize, cell_bold_style.leading),
            ("SPAN", (0, 0), (-1, 0)),
            ("SPAN", (0, 1), (1, 2)),
            ("SPAN", (2, 1), (-1, 1)),
            ("SPAN", (0, 3), (0, -1)),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ("BACKGROUND", (2, 3), (-1, -1), colors.HexColor("#4d93d9")),
        ])
    ))
    story.append(Spacer(1, 60))
    
    story.append(Table([[Paragraph(
        """
 <para>
  <b>How to Read This Chart</b><br/><br/>
  The <b>Shaw Strengths Matrix&#8482; (SSM&#8482;)</b> synthesises four interrelated frameworks into one, providing nuanced insights into your unique preference rankings. It integrates how we:<br/><br/>
  * <b>Apply our Cognitive Preferences</b> — <i>Intuition</i>, <i>Thinking</i>, <i>Feeling</i>, and <i>Sensing</i><br/>
  * <b>Express these across our Temporal Preferences</b> — <i>Past Reflections</i>, <i>Present Awareness</i>, and <i>Future Anticipations</i><br/>
  * <b>Combine these two dimensions</b> into <b>cognitive–affective units</b> which, when aggregated across multiple situations, define your ranked suite of <b>Character Strengths</b>.<br/><br/>
  Each Strength is therefore a <b>composite</b> of both <b>Cognitive Preference</b> (<i>Intuition</i>, <i>Thinking</i>, <i>Feeling</i>, and <i>Sensing</i>) and <b>Temporal Preference</b> (<i>Past Reflections</i>, <i>Present Awareness</i>, and <i>Future Anticipations</i>).<br/><br/>
  Your <b>survey responses</b> were scored and ranked using a <b>standardised algorithm</b> designed to ensure consistent comparison across all Strengths.<br/><br/> 
  These rankings reflect patterns described in the <b>CAPS model</b> (Mischel &amp; Shoda), which explains how people tend to think, feel, and behave in consistent ways across different situations. Your higher-ranked Strengths represent tendencies that are <b>more readily accessible</b> to you and  therefore guide your responses in real-world situations more often.<br/>
</para>
        """,
        style=body_style
    )]], style=table_border))
    story.append(PageBreak())

    # Page 4 - Assessment Table
    story.append(header_template(4, "Assessment Table"))
    story.append(Spacer(1, 12))

    results_table_data = [["Rank", "Category", "SSM\nStrengths™"]]
    results_table_style = [
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONT", (0, 0), (-1, -1), cell_bold_style.fontName, cell_bold_style.fontSize, cell_bold_style.leading),
        ("GRID", (0, 0), (-1, 1), 1, colors.black),
    ]
    
    last_category = None
    category_start = len(results_table_data)
    
    for idx, trait in enumerate(ordered_traits):
        rank = ranks[trait]
        category = category_for_rank_number(rank)

        if last_category != category:
            if last_category is not None:
                results_table_style.extend([
                    ("GRID", (0, category_start), (-1, len(results_table_data) - 1), 1, colors.black),
                    ("BACKGROUND", (0, category_start), (1, len(results_table_data) - 1), _color_for_category(last_category)),
                ])
                results_table_data.append(["","",""])
                results_table_style.append(("FONTSIZE", (0, len(results_table_data) - 1), (-1, len(results_table_data) - 1), 4))
                results_table_style.append(("LEADING", (0, len(results_table_data) - 1), (-1, len(results_table_data) - 1), 4))

            last_category = category
            category_start = len(results_table_data)
        results_table_data.append([_fmt_rank(rank), category, trait])

    results_table_style.extend([
        ("GRID", (0, category_start), (-1, -1), 1, colors.black),
        ("BACKGROUND", (0, category_start), (1, -1), _color_for_category(category)),
    ])

    descriptions = {"SSM\nStrengths™": "<b>Description</b>", "": ""} | DESCRIPTIONS
    story.append(Table(
        [row + [Paragraph(descriptions[row[2]], style=cell_style)] for row in results_table_data],
        style=results_table_style,
        colWidths=[0.45*inch, 1.1*inch, 1.1*inch, None],
    ))
    story.append(Spacer(1, 12))
    
    story.append(Table([[Paragraph(
        """
        <b>How to Read This Chart</b><br/>
        <br/>
        The <b>SSM™ Assessment Table</b> presents a high-level summary of the results from your completed assessment.<br/>
        It ranks the measured presence of the <b>12 SSM™ Character Strengths</b> in your personality — based on your responses to workplace scenario questions — from <b>1 (strongest)</b> to <b>12 (least expressed)</b>.<br/>
        <br/>
        Your Character Strengths are grouped into three categories:<br/>
        <br/>
        a) <b>Signature Strengths</b> – your core or defining qualities. These consistently shape how you think, feel, and act — your natural "signature moves."<br/>
        b) <b>Supporting Strengths</b> – qualities that complement your signature strengths. They are reliable and useful but not always dominant or expressed in every context.<br/>
        c) <b>Emerging Strengths</b> – qualities that are less consistently expressed or still developing. They represent areas for growth and potential to strengthen further.<br/>
        """,
        style=body_style
    )]], style=table_border))
    story.append(PageBreak())

    # Page 5 - O*NET Work Styles
    story.append(header_template(5, "Mapping to O*NET Work Styles"))
    story.append(Spacer(1, 12))
    
    work_styles = {"SSM\nStrengths™": ("Work Styles (O*NET)", "<b>Description</b>"), "": ("","")} | ONET_STYLES
    story.append(Table(
        [row + [Paragraph(work_styles[row[2]][0], style=cell_bold_center_style), Paragraph(work_styles[row[2]][1], style=cell_style)]
         for row in results_table_data],
        style=results_table_style,
        colWidths=[0.45*inch, 1.1*inch, 1.1*inch, 1.1*inch, None],
    ))
    story.append(Spacer(1, 12))
    
    story.append(Table([[Paragraph(
        """
        <b>How to Read This Chart</b><br/>
        <br/>
        This chart maps your 12 ranked <b>SSM™ Character Strengths</b> to the 12 core <b>O*NET Work Styles</b>, illustrating how your strengths translate into observable workplace behaviours.<br/>
        <br/>
        Your <b>SSM™ Assessment</b> rankings (1–12) and <b>Categories</b> (<i>Signature</i>, <i>Supporting</i>, and <i>Emerging</i>) align directly with the corresponding <b>O*NET Work Styles</b> listed here.<br/>
        <br/>
        <b>O*NET</b> defines Work Styles as "personal characteristics that can affect how well someone performs a job."<br/>
        They represent the <b>workplace expression</b> of your Character Strengths — showing how your inner traits are activated and applied in professional settings.<br/>
        <br/>
        Your <b>Work Styles</b> ranking reveals the underlying <b>"why"</b> — your motivation and natural approach to work.<br/>
        """,
        style=body_style
    )]], style=table_border))
    story.append(PageBreak())

    # Page 6 - O*NET Work Activities
    story.append(header_template(6, "Mapping to O*NET Work Activities"))
    story.append(Spacer(1, 12))
    
    activities = {"SSM\nStrengths™": "<b>Work Activities (O*NET)</b>", "": ""} | ONET_ACTIVITIES
    story.append(Table(
        [row + [Paragraph(activities[row[2]], style=cell_center_style)] for row in results_table_data],
        style=results_table_style,
        colWidths=[0.45*inch, 1.1*inch, 1.1*inch, None],
    ))
    
    story.append(PageBreak())
    story.append(Table([[
        Paragraph("Shaw Strengths Matrix™ Assessment", style=body_style),
        Paragraph(f"{first} {last} | Page 7", style=body_right_style),
    ]]))
    
    story.append(Table([[Paragraph(
        """
        <b>How to Read This Chart</b><br/>
        <br/>
        This chart maps your 12 ranked <b>SSM™ Character Strengths</b> and 12 ranked <b>O*NET Work Styles</b> to the 36 core <b>O*NET Work Activities</b>, illustrating how your strengths translate into observable task preferences.<br/>
        <br/>
        Your <b>SSM™ Assessment</b> rankings (1–12) and <b>Categories</b> (<i>Signature</i>, <i>Supporting</i>, and <i>Emerging</i>) align directly with the corresponding <b>O*NET Work Activities</b> listed here.<br/>
        <br/>
        <b>O*NET</b> defines Work Activities as "general types of job behaviours occurring on multiple jobs."<br/>
        They represent the <b>task-level expression</b> of your Character Strengths and Work Styles — showing how your inner traits and workplace behaviours manifest as more or less preferred types of tasks.<br/>
        <br/>
        Your <b>Work Activities</b> ranking defines the <b>"how"</b> — the method and style behind your approach to completing work tasks.<br/>
        """,
        style=body_style
    )]], style=table_border))

    doc.build(story)
    
    filename = f"SSM_{first}_{last}_{date_str}_v1.pdf"
    return filename


def create_distribution_chart_drawing(
    distribution_data: Dict[str, Dict[str, float]], 
    ordered_traits: List[str],
    width=500, 
    height=300
) -> Drawing:
    """
    Create a stacked bar chart showing the distribution of strength categories across the team.
    
    Args:
        distribution_data: Dict mapping trait names to category percentages
            Example: {
                "Fairness": {"Signature": 0.40, "Supporting": 0.35, "Emerging": 0.25},
                "Empathy": {"Signature": 0.30, "Supporting": 0.50, "Emerging": 0.20},
                ...
            }
        ordered_traits: List of traits in ranked order (1-12) to match team assessment table
        width: Chart width in points
        height: Chart height in points
    
    Returns:
        Drawing object containing the stacked bar chart with one bar per trait
    """
    from math import sin, cos, radians
    
    drawing = Drawing(width, height)
    
    # Use the ranked order from the team assessment table (rank 1 first, rank 12 last)
    traits = ordered_traits
    num_traits = len(traits)
    
    # Chart dimensions and positioning
    chart_left = 50
    chart_bottom = 50
    chart_width = width - 100
    chart_height = height - 100
    
    # Bar dimensions
    bar_width = chart_width / (num_traits * 1.5)  # Leave space between bars
    bar_spacing = bar_width * 0.5
    
    # Draw Y-axis gridlines and labels
    for i in range(0, 101, 20):
        y_pos = chart_bottom + (i / 100.0) * chart_height
        # Gridline label
        drawing.add(String(chart_left - 5, y_pos - 3, f"{i}%", fontSize=8, textAnchor='end'))
        # Horizontal grid line
        drawing.add(Line(chart_left, y_pos, chart_left + chart_width, y_pos, strokeColor=colors.lightgrey, strokeWidth=0.5))
    
    # Draw stacked bars for each trait
    for i, trait in enumerate(traits):
        # Calculate bar x position
        x_pos = chart_left + i * (bar_width + bar_spacing)
        
        # Get percentages for this trait
        signature_pct = distribution_data[trait].get("Signature", 0)
        supporting_pct = distribution_data[trait].get("Supporting", 0)
        emerging_pct = distribution_data[trait].get("Emerging", 0)
        
        # Calculate heights (in points)
        signature_height = signature_pct * chart_height
        supporting_height = supporting_pct * chart_height
        emerging_height = emerging_pct * chart_height
        
        # Draw Signature segment (bottom) - dark blue
        drawing.add(Rect(
            x_pos, chart_bottom,
            bar_width, signature_height,
            fillColor=colors.HexColor("#4d93d9"),
            strokeColor=colors.black,
            strokeWidth=1
        ))
        
        # Draw Supporting segment (middle) - light blue
        drawing.add(Rect(
            x_pos, chart_bottom + signature_height,
            bar_width, supporting_height,
            fillColor=colors.HexColor("#94dcf8"),
            strokeColor=colors.black,
            strokeWidth=1
        ))
        
        # Draw Emerging segment (top) - grey
        drawing.add(Rect(
            x_pos, chart_bottom + signature_height + supporting_height,
            bar_width, emerging_height,
            fillColor=colors.HexColor("#d0d0d0"),
            strokeColor=colors.black,
            strokeWidth=1
        ))
        
        # Add trait label below bar (rotated 45 degrees)
        label_x = x_pos + bar_width / 2
        label_y = chart_bottom - 10
        
        label_group = Group()
        label_text = String(0, 0, trait, fontSize=8, textAnchor='end')
        label_group.add(label_text)
        
        # Apply 45-degree rotation transform
        angle = 45
        rad = radians(angle)
        label_group.transform = (cos(rad), sin(rad), -sin(rad), cos(rad), label_x, label_y)
        drawing.add(label_group)
    
    # Add legend - centered horizontally above the chart
    legend_y = chart_bottom + chart_height + 20  # Position above chart
    
    # Calculate total legend width to center it
    # Each item: 12 (rect) + 5 (space) + ~60 (text) + 15 (spacing) = ~92 per item
    legend_total_width = 3 * 90  # Approximate width for 3 legend items
    legend_start_x = (width - legend_total_width) / 2  # Center horizontally
    
    # Signature legend
    drawing.add(Rect(legend_start_x, legend_y, 12, 12, fillColor=colors.HexColor("#4d93d9"), strokeColor=colors.black))
    drawing.add(String(legend_start_x + 15, legend_y + 3, "Signature", fontSize=9))
    
    # Supporting legend
    supporting_x = legend_start_x + 90
    drawing.add(Rect(supporting_x, legend_y, 12, 12, fillColor=colors.HexColor("#94dcf8"), strokeColor=colors.black))
    drawing.add(String(supporting_x + 15, legend_y + 3, "Supporting", fontSize=9))
    
    # Emerging legend
    emerging_x = supporting_x + 90
    drawing.add(Rect(emerging_x, legend_y, 12, 12, fillColor=colors.HexColor("#d0d0d0"), strokeColor=colors.black))
    drawing.add(String(emerging_x + 15, legend_y + 3, "Emerging", fontSize=9))
    
    return drawing


def calculate_team_rankings(
    individual_results: List[Dict[str, Any]]
) -> Tuple[List[str], Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    Calculate team-level rankings by averaging individual rankings.
    
    Algorithm steps:
    1. Calculate average ranks across all team members
    2. Sort traits by average rank
    2.1. Convert ranked averages to 1-12 rankings
    2.2. Apply median tie-breaker for traits with same mean rank
    2.3. Apply average ranking for remaining ties (same mean and median)
    3. Calculate distribution data (percentage in each category)
    
    Args:
        individual_results: List of dicts containing 'ordered_traits' and 'ranks' for each person
    
    Returns:
        Tuple of (ordered_traits, ranks, distribution_data)
            - ordered_traits: List of traits ordered by team average rank
            - ranks: Dict mapping trait to final team rank (1-12, may include .5 for ties)
            - distribution_data: Dict mapping trait to category distribution
    """
    from statistics import median
    
    # Step 1: Calculate average ranks and store individual ranks for median calculation
    rank_sums = {trait: 0.0 for trait in TRAITS}
    rank_counts = {trait: 0 for trait in TRAITS}
    individual_ranks_by_trait = {trait: [] for trait in TRAITS}
    
    for result in individual_results:
        for trait, rank in result['ranks'].items():
            rank_sums[trait] += rank
            rank_counts[trait] += 1
            individual_ranks_by_trait[trait].append(rank)
    
    avg_ranks = {
        trait: rank_sums[trait] / rank_counts[trait] if rank_counts[trait] > 0 else 12
        for trait in TRAITS
    }
    
    # Step 2: Sort traits by average rank (initial ordering)
    ordered_traits_temp = sorted(TRAITS, key=lambda t: (avg_ranks[t], t))
    
    # Step 2.1 & 2.2: Group by mean rank, then apply median tie-breaker
    # Group traits with the same mean rank
    trait_groups = []
    i = 0
    while i < len(ordered_traits_temp):
        current_trait = ordered_traits_temp[i]
        current_avg = avg_ranks[current_trait]
        
        # Find all traits with same average rank (within floating point tolerance)
        group = [current_trait]
        j = i + 1
        while j < len(ordered_traits_temp):
            next_trait = ordered_traits_temp[j]
            if abs(avg_ranks[next_trait] - current_avg) < 0.0001:  # Floating point tolerance
                group.append(next_trait)
                j += 1
            else:
                break
        
        # Step 2.2: If group has multiple traits (tie), sort by median
        if len(group) > 1:
            median_ranks = {
                trait: median(individual_ranks_by_trait[trait]) if individual_ranks_by_trait[trait] else 12
                for trait in group
            }
            # Sort by median, then alphabetically for stability
            group.sort(key=lambda t: (median_ranks[t], t))
        
        trait_groups.append(group)
        i = j
    
    # Flatten groups back into ordered list
    ordered_traits_with_median = []
    for group in trait_groups:
        ordered_traits_with_median.extend(group)
    
    # Step 2.3: Assign final ranks with average ranking for remaining ties
    # Group again by (mean, median) to find remaining ties
    final_ranks = {}
    nominal_position = 1
    
    i = 0
    while i < len(ordered_traits_with_median):
        current_trait = ordered_traits_with_median[i]
        current_avg = avg_ranks[current_trait]
        current_median = median(individual_ranks_by_trait[current_trait]) if individual_ranks_by_trait[current_trait] else 12
        
        # Find all traits with same average AND median
        tie_group = [current_trait]
        j = i + 1
        while j < len(ordered_traits_with_median):
            next_trait = ordered_traits_with_median[j]
            next_avg = avg_ranks[next_trait]
            next_median = median(individual_ranks_by_trait[next_trait]) if individual_ranks_by_trait[next_trait] else 12
            
            if abs(next_avg - current_avg) < 0.0001 and abs(next_median - current_median) < 0.0001:
                tie_group.append(next_trait)
                j += 1
            else:
                break
        
        # Assign average of nominal positions to all traits in tie group
        if len(tie_group) == 1:
            # No tie, assign nominal position
            final_ranks[current_trait] = float(nominal_position)
        else:
            # Tie: assign average of nominal positions
            nominal_positions = list(range(nominal_position, nominal_position + len(tie_group)))
            avg_position = mean(nominal_positions)
            for trait in tie_group:
                final_ranks[trait] = avg_position
        
        # Move to next group, skipping positions used by tie group
        nominal_position += len(tie_group)
        i = j
    
    # Step 2.1 (final): ordered_traits now reflects the final ranking order
    ordered_traits = ordered_traits_with_median
    
    # Step 3: Calculate distribution data (percentage in each category)
    # This uses the ORIGINAL individual ranks, not the team average ranks
    distribution_data = {}
    total_count = len(individual_results)
    
    for trait in TRAITS:
        category_counts = {"Signature": 0, "Supporting": 0, "Emerging": 0}
        
        for result in individual_results:
            rank = result['ranks'].get(trait, 12)
            category = category_for_rank_number(rank)
            category_counts[category] += 1
        
        # Convert to percentages
        distribution_data[trait] = {
            cat: count / total_count if total_count > 0 else 0
            for cat, count in category_counts.items()
        }
    
    return ordered_traits, final_ranks, distribution_data


def generate_team_pdf(
    company_name: str,
    team_name: str,
    num_members: int,
    date_str: str,
    ordered_traits: List[str],
    ranks: Dict[str, float],
    distribution_data: Dict[str, Dict[str, float]],
    output_stream: io.BytesIO,
    logo_path: str = LOGO_PATH,
) -> str:
    """
    Generate team PDF report and write to output_stream.
    Returns the suggested filename.
    """
    
    def _fmt_rank(r):
        try:
            r = float(r)
            return str(int(r)) if r.is_integer() else f"{r:.1f}"
        except Exception:
            return str(r)
    
    def _color_for_category(cat: str):
        return colors.HexColor(
            "#4d93d9" if cat == "Signature" else (
            "#94dcf8" if cat == "Supporting" else
            "#d0d0d0")
        )
    
    date = _parse_date_long(date_str)
    
    doc = SimpleDocTemplate(
        output_stream, pagesize=A4, leftMargin=40, rightMargin=40, topMargin=36, bottomMargin=36
    )
    styles = getSampleStyleSheet()
    
    header_style = ParagraphStyle(
        "SSMHeader", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=22, leading=27, alignment=TA_LEFT, spaceAfter=8
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"], fontName="Helvetica",
        fontSize=11, leading=13, alignment=TA_LEFT
    )
    body_bold_style = ParagraphStyle("CellBold", parent=body_style, fontName="Helvetica-Bold")
    body_right_style = ParagraphStyle("AsideRight", parent=body_style, alignment=TA_RIGHT)
    cell_style = ParagraphStyle("Cell", parent=body_style, fontSize=9, leading=10)
    cell_bold_style = ParagraphStyle("CellBold", parent=cell_style, fontName="Helvetica-Bold")
    cell_center_style = ParagraphStyle("CellCenter", parent=cell_style, alignment=TA_CENTER)
    cell_bold_center_style = ParagraphStyle("CellBoldCenter", parent=cell_bold_style, alignment=TA_CENTER)
    # Compact style for Work Activities table to fit on one page
    cell_compact_style = ParagraphStyle("CellCompact", parent=body_style, fontSize=7.5, leading=8.5, alignment=TA_CENTER)
    
    table_border = TableStyle([
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
    ])
    
    story = []
    
    # Page 1 - Title page
    story.append(Table(
        [[Paragraph("Shaw Strengths Matrix™ (SSM™)<br/>Assessment", header_style)]],
        style=table_border
    ))
    
    story.append(Spacer(1, 6))
    
    story.append(Table([[Paragraph("", style=body_style)]], style=TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#ba9354"))
    ])))
    
    story.append(Table([
        [Paragraph("Report prepared for:", style=body_bold_style)],
        [Paragraph(f"Organisation: {company_name}", style=body_bold_style)],
        [Paragraph(f"Team: {team_name}", style=body_bold_style)],
        [Paragraph(f"Number of Team members: {num_members}", style=body_bold_style)],
        [Paragraph(f"Date: {date: %d %B %Y}", style=body_bold_style)],
        ],
        style=table_border,
    ))
    story.append(Spacer(1, 78))
    
    if os.path.exists(logo_path):
        logo_img = Image(logo_path, width=1.54*inch, height=0.79*inch, kind="proportional")
    else:
        logo_img = Paragraph("", styles["Normal"])
    
    story.append(Table([[
        logo_img,
        Paragraph("ShawSight Pty Ltd   |   ABN:  38688414557", style=body_style),
    ]]))
    story.append(Spacer(1,6))
    
    legal_notices = Paragraph(
        """
        <i>Shaw Strengths Matrix™ Framework/i> Copyright 2025 by ShawSight Pty Ltd. All rights reserved.<br/>
        <i>Shaw Strengths Matrix™ Assessment/i> Copyright 2025 by ShawSight Pty Ltd. All rights reserved.<br/>
        <i>ShawSight</i> logo is Copyright 2025 by ShawSight Pty Ltd. All rights reserved.<br/>
        No part of this publication may be reproduced in any form or manner without prior written permission from ShawSight Pty Ltd.<br/>
        O*NET is a trademark of the U.S. Department of Labor, Employment and Training Administration.""",
        style=body_style
    )
    story.append(Table([[legal_notices]], style=table_border))
    
    story.append(PageBreak())
    
    # Header template for pages 2-7
    def header_template(page_num: int, subtitle: str) -> Table:
        return Table(
            [[
                Paragraph("Shaw Strengths Matrix™ Assessment", style=body_style),
                Paragraph(f"{team_name} | Page {page_num}", style=body_right_style),
            ], [
                Paragraph(f"Shaw Strengths Matrix™<br/>{subtitle}", style=header_style),
                ""
            ]],
            style=TableStyle([
                ("BOX", (0, 1), (1, 1), 1, colors.black),
                ("SPAN", (0, 1), (1, 1)),
            ]),
        )
    
    # Page 2 - Overview
    story.append(header_template(2, "Overview"))
    story.append(Spacer(1, 6))
    
    story.append(Table([[Paragraph(
        """
        <b>About This Report</b><br/>
        <br/>
        This report provides an overview of the team's <b>SSM™ Assessment Character Strengths</b> profile and how it may relate to patterns of <b>communication, collaboration, and overall team dynamics</b> at work. It is designed to support <b>self-awareness, reflection, and developmental discussion</b> in team and workshop settings.<br/>
        <br/>
        The <b>SSM™ Assessment</b> measures <b>personality</b> — how the team collectively tends to behave — rather than <b>abilities or skills</b> (what you are good at) or <b>interests</b> (what you enjoy doing). It is <b>not</b> intended to provide any clinical diagnosis.<br/>
        <br/>
        While the Assessment is grounded in established <b>personality and work style research</b> and has passed <b>content and face validity testing</b>, it is currently undergoing further psychometric validation. Results should therefore be interpreted as <b>insightful tendencies</b> rather than predictive measures, and are <b>not intended</b> for hiring, promotion, or other HR decision-making.<br/>
        <br/>
        In this report, your <b>SSM™ Assessment</b> Character Strengths profile is also conceptually aligned to <b>O*NET Work Styles</b> and <b>Work Activities</b>.<br/>
        <br/>
        The <b>O*NET Resource Center</b> is a professional workforce research portal providing data, tools, technical documentation, and support. It is widely recognised as a <b>global standard in workplace metrics</b>.
        """,
        style=body_style
    )]], style=table_border))
    story.append(Spacer(1, 6))
    
    story.append(Table([[Paragraph(
        """
        <b>Report Contents</b><br/><br/>
        1) Shaw Strengths Matrix™<br/>
        2) Shaw Strengths Matrix™ Team Assessment Table<br/>
        3) Shaw Strengths Matrix™ Team Mapping to O*NET Work Styles<br/>
        4) Shaw Strengths Matrix™ Team Mapping to O*NET Work Activities<br/>
        5) Shaw Strengths Matrix™ Team Distribution Chart of Strength Categories
        """,
        style=body_style
    )]], style=table_border))
    story.append(PageBreak())
    
    # Page 3 - Matrix explanation
    story.append(header_template(3, ""))
    story.append(Spacer(1, 12))
    
    story.append(Table(
        [
            ["Shaw Strengths Matrix™","","","",""],
            ["Character Strengths","","Temporal Preferences","",""],
            ["","", "Past Reflections", "Present Awareness", "Future Anticipations"],
            ["Cognitive\nPreferences", "Intuition", "Foresight", "Confidence", "Courage"],
            ["", "Thinking", "Curiosity", "Objectivity", "Fairness"],
            ["", "Feeling", "Empathy", "Tenacity", "Prudence"],
            ["", "Sensing", "Discernment", "Practicality", "Discipline"]
        ],
        style=TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("FONT", (0, 0), (-1, -1), cell_bold_style.fontName, cell_bold_style.fontSize, cell_bold_style.leading),
            ("SPAN", (0, 0), (-1, 0)),
            ("SPAN", (0, 1), (1, 2)),
            ("SPAN", (2, 1), (-1, 1)),
            ("SPAN", (0, 3), (0, -1)),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ("BACKGROUND", (2, 3), (-1, -1), colors.HexColor("#4d93d9")),
        ])
    ))
    story.append(Spacer(1, 60))
    
    story.append(Table([[Paragraph(
        """
 <para>
  <b>How to Read This Chart</b><br/><br/>
  The <b>Shaw Strengths Matrix&#8482; (SSM&#8482;)</b> synthesises four interrelated frameworks into one, providing nuanced insights into individual unique preference rankings. It integrates how we:<br/><br/>
  * <b>Apply our Cognitive Preferences</b> — <i>Intuition</i>, <i>Thinking</i>, <i>Feeling</i>, and <i>Sensing</i> —<br/>
  * <b>Express these across our Temporal Preferences</b> — <i>Past Reflections</i>, <i>Present Awareness</i>, and <i>Future Anticipations</i>, and<br/>
  * <b>Combines these two dimensions</b> into <b>cognitive–affective units</b> which, when aggregated across multiple situations, define your ranked suite of <b>Character Strengths</b>.<br/><br/>
  Each Strength is therefore a <b>composite</b> of both <b>Cognitive Preference</b> (<i>Intuition</i>, <i>Thinking</i>, <i>Feeling</i>, and <i>Sensing</i>) and <b>Temporal Preference</b> (<i>Past Reflections</i>, <i>Present Awareness</i>, and <i>Future Anticipations</i>).<br/><br/>
  Individual survey responses were scored and ranked using a <b>standardised algorithm</b> designed to ensure consistent comparison across all Strengths.<br/><br/> 
  These rankings reflect patterns described in the <b>CAPS model</b> (Mischel &amp; Shoda), which explains how people tend to think, feel, and behave in consistent ways across different situations. Our higher-ranked Strengths represent tendencies that are <b>more readily accessible</b> to us and therefore guide our responses in real-world situations more often.<br/>
</para>
        """,
        style=body_style
    )]], style=table_border))
    story.append(PageBreak())
    
    # Page 4 - Team Assessment Table
    story.append(header_template(4, "Team Assessment Table"))
    story.append(Spacer(1, 12))
    
    results_table_data = [["Rank", "Category", "SSM\nStrengths™"]]
    results_table_style = [
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONT", (0, 0), (-1, -1), cell_bold_style.fontName, cell_bold_style.fontSize, cell_bold_style.leading),
        ("GRID", (0, 0), (-1, 1), 1, colors.black),
    ]
    
    last_category = None
    category_start = len(results_table_data)
    
    for idx, trait in enumerate(ordered_traits):
        rank = ranks[trait]
        category = category_for_rank_number(rank)
        
        if last_category != category:
            if last_category is not None:
                results_table_style.extend([
                    ("GRID", (0, category_start), (-1, len(results_table_data) - 1), 1, colors.black),
                    ("BACKGROUND", (0, category_start), (1, len(results_table_data) - 1), _color_for_category(last_category)),
                ])
                results_table_data.append(["","",""])
                results_table_style.append(("FONTSIZE", (0, len(results_table_data) - 1), (-1, len(results_table_data) - 1), 4))
                results_table_style.append(("LEADING", (0, len(results_table_data) - 1), (-1, len(results_table_data) - 1), 4))
            
            last_category = category
            category_start = len(results_table_data)
        results_table_data.append([_fmt_rank(rank), category, trait])
    
    results_table_style.extend([
        ("GRID", (0, category_start), (-1, -1), 1, colors.black),
        ("BACKGROUND", (0, category_start), (1, -1), _color_for_category(category)),
    ])
    
    descriptions = {"SSM\nStrengths™": "<b>Description</b>", "": ""} | DESCRIPTIONS
    story.append(Table(
        [row + [Paragraph(descriptions[row[2]], style=cell_style)] for row in results_table_data],
        style=results_table_style,
        colWidths=[0.45*inch, 1.1*inch, 1.1*inch, None],
    ))
    story.append(Spacer(1, 12))
    
    story.append(Table([[Paragraph(
        """
        <b>How to Read This Chart</b><br/>
        <br/>
        The <b>SSM™ Assessment Table</b> presents a high-level summary of the results from the averaged rankings across the team for all included assessments.<br/>
        It ranks the measured presence of the <b>12 SSM™ Character Strengths</b> in your personality — based on the team's collective responses to workplace scenario questions — from <b>1 (strongest)</b> to <b>12 (least expressed)</b>.<br/>
        <br/>
        The team's collective Character Strengths are grouped into three categories:<br/>
        <br/>
        a) <b>Signature Strengths</b> – the team's core or defining qualities. These consistently shape how team members collectively think, feel, and act — the team's natural "signature moves."<br/>
        b) <b>Supporting Strengths</b> – qualities that complement the team's signature strengths. They are reliable and useful but not always dominant or expressed in every context.<br/>
        c) <b>Emerging Strengths</b> – qualities that are less consistently expressed or still developing. They represent areas for growth and potential for the team to strengthen further.<br/>
        """,
        style=body_style
    )]], style=table_border))
    story.append(PageBreak())
    
    # Page 5 - O*NET Work Styles
    story.append(header_template(5, "Team Mapping to O*NET Work Styles"))
    story.append(Spacer(1, 12))
    
    work_styles = {"SSM\nStrengths™": ("Work Styles (O*NET)", "<b>Description</b>"), "": ("","")} | ONET_STYLES
    story.append(Table(
        [row + [Paragraph(work_styles[row[2]][0], style=cell_bold_center_style), Paragraph(work_styles[row[2]][1], style=cell_style)]
         for row in results_table_data],
        style=results_table_style,
        colWidths=[0.45*inch, 1.1*inch, 1.1*inch, 1.1*inch, None],
    ))
    story.append(Spacer(1, 12))
    
    story.append(Table([[Paragraph(
        """
        <b>How to Read This Chart</b><br/>
        <br/>
        This chart maps the team's 12 ranked <b>SSM™ Character Strengths</b> to the 12 core <b>O*NET Work Styles</b>, illustrating how overall team strengths translate into observable workplace behaviours.<br/>
        <br/>
        The team <b>SSM™ Assessment</b> rankings (1–12) and <b>Categories</b> (<i>Signature</i>, <i>Supporting</i>, and <i>Emerging</i>) align directly with the corresponding <b>O*NET Work Styles</b> listed here.<br/>
        <br/>
        <b>O*NET</b> defines Work Styles as "personal characteristics that can affect how well someone performs a job."<br/>
        They represent the <b>workplace expression</b> of your Character Strengths — showing how your inner traits are activated and applied in professional settings.<br/>
        <br/>
        The team <b>Work Styles</b> ranking reveals the underlying <b>"why"</b> — the team's overall motivation and natural approach to work.<br/>
        """,
        style=body_style
    )]], style=table_border))
    story.append(PageBreak())
    
    # Page 6 - O*NET Work Activities
    story.append(header_template(6, "Team Mapping to O*NET Work Activities"))
    story.append(Spacer(1, 8))  # Reduced from 12 to 8
    
    # Create compact table style with reduced padding
    compact_table_style = list(results_table_style) + [
        ("TOPPADDING", (0, 0), (-1, -1), 2),    # Reduced padding
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ]
    
    activities = {"SSM\nStrengths™": "<b>Work Activities (O*NET)</b>", "": ""} | ONET_ACTIVITIES
    story.append(Table(
        [row + [Paragraph(activities[row[2]], style=cell_compact_style)] for row in results_table_data],
        style=compact_table_style,
        colWidths=[0.4*inch, 1.0*inch, 1.0*inch, None],  # Slightly reduced column widths
    ))
    story.append(Spacer(1, 6))  # Reduced spacing before explanation
    
    # Create compact body style for this explanation
    body_compact_style = ParagraphStyle("BodyCompact", parent=body_style, fontSize=9.5, leading=11)
    
    story.append(Table([[Paragraph(
        """
        <b>How to Read This Chart</b><br/>
        <br/>
        This chart maps the team's 12 ranked <b>SSM™ Character Strengths</b> and 12 ranked <b>O*NET Work Styles</b> to the 36 core <b>O*NET Work Activities</b>, illustrating how overall team strengths translate into observable task preferences.<br/>
        <br/>
        The team <b>SSM™ Assessment</b> rankings (1–12) and <b>Categories</b> (<i>Signature</i>, <i>Supporting</i>, and <i>Emerging</i>) align directly with the corresponding <b>O*NET Work Activities</b> listed here.<br/>
        <br/>
        <b>O*NET</b> defines Work Activities as "general types of job behaviours occurring on multiple jobs."<br/>
        They represent the <b>task-level expression</b> of your Character Strengths and Work Styles — showing how your inner traits and workplace behaviours manifest as more or less preferred types of tasks.<br/>
        <br/>
        The team <b>Work Activities</b> ranking defines the <b>"how"</b> — the method and style behind the team's overall approach to completing work tasks.<br/>
        """,
        style=body_compact_style
    )]], style=table_border))
    
    # Page 7 - Team Distribution Chart
    story.append(PageBreak())
    story.append(Table([[
        Paragraph("Shaw Strengths Matrix™ Assessment", style=body_style),
        Paragraph(f"{team_name} | Page 7", style=body_right_style),
    ]]))
    story.append(Spacer(1, 6))
    
    story.append(Table([[Paragraph(
        "Shaw Strengths Matrix™<br/>Team Distribution Chart of Strength Categories",
        style=header_style
    )]], style=table_border))
    story.append(Spacer(1, 12))
    
    # Add distribution chart (ordered by team ranking: rank 1 first, rank 12 last)
    chart_drawing = create_distribution_chart_drawing(distribution_data, ordered_traits, width=500, height=300)
    story.append(chart_drawing)
    story.append(Spacer(1, 12))
    
    story.append(Table([[Paragraph(
        """
        <b>How to Read This Chart</b><br/>
        <br/>
        Each bar represents one of the 12 SSM™ Character Strengths.<br/>
        <br/>
        It is divided into three colour segments — Signature, Supporting, and Emerging — showing what percentage of the team placed the strength in each category.<br/>
        Every bar totals 100%, making it easy to compare strengths side by side.<br/>
        <br/>
        * <b>Signature segments</b> show the team's strongest instincts.<br/>
        A tall Signature portion means many team members naturally rely on that strength. These usually indicate areas of high capability and team advantage.<br/>
        <br/>
        * <b>Supporting segments</b> represent moderate, flexible strengths.<br/>
        These are strengths the team can use when needed, but which are not core drivers. A tall Supporting segment means many team members placed that strength in the mid-range.<br/>
        <br/>
        * <b>Emerging segments</b> highlight development areas.<br/>
        A tall Emerging portion means few team members prioritise or identify strongly with that strength. These may indicate capability gaps or growth opportunities.<br/>
        <br/>
        Taken together, these patterns help compare the team's current profile with a desired profile for the role, department, or organisation.<br/>
        They reveal where the team is well-aligned and where development or rebalancing may be beneficial.<br/>
        """,
        style=body_style
    )]], style=table_border))
    
    doc.build(story)
    
    filename = f"SSM_Team_{company_name}_{team_name}_{date_str}_v1.pdf"
    return filename


def process_excel_to_pdf(excel_bytes: bytes, original_filename: str = "assessment.xlsx") -> Tuple[bytes, str]:
    """
    Process Excel file bytes and return PDF bytes and filename.
    
    Args:
        excel_bytes: Raw bytes of the Excel file
        original_filename: Original filename (for logging)
    
    Returns:
        Tuple of (pdf_bytes, pdf_filename)
    """
    # Write Excel to temporary file for pandas to read
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp_excel:
        tmp_excel.write(excel_bytes)
        tmp_excel_path = tmp_excel.name

    try:
        # Extract identity and parse survey
        first, last, date_str = extract_identity_flexible(tmp_excel_path)
        results = parse_survey_flexible(tmp_excel_path)

        # Calculate rankings
        tb = TieBreaker(TRAITS, results)
        ordered_traits, ranks = tb.rank_with_average_ranks()

        # Generate PDF to memory
        pdf_buffer = io.BytesIO()
        pdf_filename = generate_pdf(first, last, date_str, ordered_traits, ranks, pdf_buffer, LOGO_PATH)
        
        pdf_bytes = pdf_buffer.getvalue()
        return pdf_bytes, pdf_filename

    finally:
        # Clean up temp file
        os.unlink(tmp_excel_path)


# ---------------------------
# FastAPI Application
# ---------------------------
app = FastAPI(
    title="SSM PDF Generator",
    description="Shaw Strengths Matrix™ PDF Generator - Cloud Run Service",
    version="1.0.0"
)

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class GeneratePDFRequest(BaseModel):
    excel_base64: str
    filename: Optional[str] = "assessment.xlsx"


class GeneratePDFResponse(BaseModel):
    success: bool
    pdf_base64: Optional[str] = None
    filename: Optional[str] = None
    message: Optional[str] = None


class GenerateTeamPDFRequest(BaseModel):
    company_name: str
    team_name: str
    num_members: int
    date_str: str  # YYYY-MM-DD
    individual_results: List[Dict[str, Any]]  # List of {'ordered_traits': [...], 'ranks': {...}}


class GenerateTeamPDFResponse(BaseModel):
    success: bool
    pdf_base64: Optional[str] = None
    filename: Optional[str] = None
    message: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for Cloud Run."""
    return HealthResponse(
        status="healthy",
        service="ssm-pdf-generator",
        version="1.0.0"
    )


@app.post("/generate-pdf-base64", response_model=GeneratePDFResponse)
async def generate_pdf_base64(request: GeneratePDFRequest):
    """
    Generate PDF from base64-encoded Excel file.
    
    Request body:
    - excel_base64: Base64-encoded Excel file
    - filename: Optional original filename
    
    Returns:
    - success: Whether PDF generation succeeded
    - pdf_base64: Base64-encoded PDF file
    - filename: Suggested filename for the PDF
    """
    try:
        # Decode base64 to bytes
        try:
            excel_bytes = base64.b64decode(request.excel_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 encoding: {str(e)}")
        
        # Process Excel and generate PDF
        pdf_bytes, pdf_filename = process_excel_to_pdf(excel_bytes, request.filename)
        
        # Encode PDF to base64
        pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
        
        return GeneratePDFResponse(
            success=True,
            pdf_base64=pdf_base64,
            filename=pdf_filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        return GeneratePDFResponse(
            success=False,
            message=f"PDF generation failed: {str(e)}"
        )


@app.post("/generate-team-pdf", response_model=GenerateTeamPDFResponse)
async def generate_team_pdf_endpoint(request: GenerateTeamPDFRequest):
    """
    Generate team PDF report from individual assessment results.
    
    Request body:
    - company_name: Company/organization name
    - team_name: Team name
    - num_members: Number of team members
    - date_str: Report date (YYYY-MM-DD)
    - individual_results: List of individual assessment results
    
    Returns:
    - success: Whether PDF generation succeeded
    - pdf_base64: Base64-encoded PDF file
    - filename: Suggested filename for the PDF
    """
    try:
        # Calculate team rankings from individual results
        ordered_traits, ranks, distribution_data = calculate_team_rankings(request.individual_results)
        
        # Generate PDF to memory
        pdf_buffer = io.BytesIO()
        pdf_filename = generate_team_pdf(
            company_name=request.company_name,
            team_name=request.team_name,
            num_members=request.num_members,
            date_str=request.date_str,
            ordered_traits=ordered_traits,
            ranks=ranks,
            distribution_data=distribution_data,
            output_stream=pdf_buffer,
            logo_path=LOGO_PATH
        )
        
        pdf_bytes = pdf_buffer.getvalue()
        
        # Encode PDF to base64
        pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
        
        return GenerateTeamPDFResponse(
            success=True,
            pdf_base64=pdf_base64,
            filename=pdf_filename
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return GenerateTeamPDFResponse(
            success=False,
            message=f"Team PDF generation failed: {str(e)}"
        )


@app.post("/generate-pdf")
async def generate_pdf_file(file: UploadFile = File(...)):
    """
    Generate PDF from uploaded Excel file.
    Returns the PDF file directly.
    
    Form data:
    - file: Excel file upload (.xlsx)
    
    Returns: PDF file download
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file selected")
        
        if not file.filename.endswith('.xlsx'):
            raise HTTPException(status_code=400, detail="File must be an Excel file (.xlsx)")
        
        # Read file bytes
        excel_bytes = await file.read()
        
        # Process and generate PDF
        pdf_bytes, pdf_filename = process_excel_to_pdf(excel_bytes, file.filename)
        
        # Return PDF file
        return Response(
            content=pdf_bytes,
            media_type='application/pdf',
            headers={
                'Content-Disposition': f'attachment; filename="{pdf_filename}"'
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")


if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get('PORT', 8080))
    uvicorn.run(app, host='127.0.0.1', port=port)
