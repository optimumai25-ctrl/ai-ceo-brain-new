# meeting_indexer.py
import re
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

PARSED_DIR = Path("parsed_data")
INDEX_JSON = PARSED_DIR / "_meeting_index.json"
INDEX_CSV = PARSED_DIR / "_meeting_index.csv"

DATE_PATTERNS = [
    "%B %d, %Y",    # September 02, 2025
    "%B %d",        # September 02
    "%Y-%m-%d",     # 2025-09-02
    "%d-%m-%Y",     # 02-09-2025
    "%d/%m/%Y",     # 02/09/2025
    "%m/%d/%Y",     # 09/02/2025
]

def _strip(s: str) -> str:
    return (s or "").strip()

def _norm_space(s: str) -> str:
    return re.sub(r"[ \t]+", " ", s).strip()

def _parse_time_date(line: str) -> (Optional[str], Optional[str]):
    # Example: "Time & Date: 4:30 AM  – September 02"
    m = re.search(r"Time\s*&\s*Date\s*:\s*(?P<time>[^–\-|]+?)[–\-]\s*(?P<date>.+)$", line, flags=re.I)
    if not m:
        return None, None
    time = _norm_space(m.group("time"))
    date = _norm_space(m.group("date"))
    return time, date

def _extract_section(text: str, title: str) -> str:
    # Capture from "^\d+\. <title>" up to next "^\d+\."
    pat = re.compile(rf"(?mi)^\s*\d+\.\s*{re.escape(title)}\s*\n(?P<body>.*?)(?=^\s*\d+\.\s*\S)", re.DOTALL)
    m = pat.search(text + "\n0.\n")  # sentinel
    return m.group("body").strip() if m else ""

def _lines_of_points(section_body: str) -> List[str]:
    # Merge wrapped lines under numbered bullets (4.1, 5.2, etc.)
    lines = [l.rstrip() for l in section_body.splitlines() if _strip(l)]
    merged, cur = [], ""
    for ln in lines:
        if re.match(r"^\s*\d+(?:\.\d+)*\s+", ln):
            if cur:
                merged.append(cur.strip())
            cur = ln
        else:
            cur += " " + ln.strip()
    if cur:
        merged.append(cur.strip())
    return merged

def _split_names(names_str: str) -> List[str]:
    # Split "Andrey & Q" / "May and Kavya" / "May, Kavya and Ed"
    s = re.sub(r"[–\-]+.*$", "", names_str)  # remove trailing task
    s = s.replace("&", " and ")
    parts = re.split(r"\s*(?:,|and)\s*", s)
    out = []
    for p in parts:
        p = p.strip()
        if p and re.match(r"^[A-Z][\w.'-]{1,}$", p):
            out.append(p)
    return list(dict.fromkeys(out))

def _extract_actions_from_points(points: List[str]) -> List[Dict]:
    """
    Extract assignee(s) and task from bullets like:
    "4.1 Sai – Update Excel ..."
    """
    actions = []
    dash = r"(?:–|-|—)"  # en/em dash or hyphen
    for pt in points:
        m = re.match(rf"^\s*\d+(?:\.\d+)*\s+([A-Z][\w.'-]+(?:\s*(?:,|and|&)\s*[A-Z][\w.'-]+)*)\s*{dash}\s*(.+)$", pt)
        if m:
            names = _split_names(m.group(1))
            task = m.group(2).strip()
            if names and task:
                actions.append({"assignees": names, "task": task, "raw": pt})
                continue
        actions.append({"assignees": [], "task": pt, "raw": pt})
    return actions

def _find_names_anywhere(text: str) -> List[str]:
    stop = {"AI","AGI","OBBA","SPV","CEO","Q3","HR","India","Visa","Team"}
    cand = set()
    for w in re.findall(r"\b([A-Z][a-z]{1,})\b", text):
        if w not in stop:
            cand.add(w)
    return sorted(cand)

def _date_from_filename(name: str) -> Optional[str]:
    m = re.search(r"(\d{4}-\d{2}-\d{2})", name)
    return m.group(1) if m else None

def parse_meeting_text(full_text: str, filename: str) -> Dict:
    lines = full_text.splitlines()
    time, date_str = None, None
    for ln in lines[:10]:  # header band
        t, d = _parse_time_date(ln)
        if d:
            time, date_str = t, d
            break

    file_date = _date_from_filename(filename)
    meeting_date_iso = None
    if date_str:
        for fmt in DATE_PATTERNS:
            try:
                dt = datetime.strptime(date_str.replace("Sept ", "September "), fmt)
                if "%Y" not in fmt and file_date:
                    dt = dt.replace(year=int(file_date.split("-")[0]))
                meeting_date_iso = dt.strftime("%Y-%m-%d")
                break
            except Exception:
                pass
    if not meeting_date_iso and file_date:
        meeting_date_iso = file_date

    next_steps_body = _extract_section(full_text, "Next Steps")
    action_items_body = _extract_section(full_text, "Action Items")

    points = _lines_of_points(next_steps_body) + _lines_of_points(action_items_body)
    actions = _extract_actions_from_points(points)

    attendees = set()
    for a in actions:
        for n in a["assignees"]:
            attendees.add(n)
        with_names = re.findall(r"\bwith\s+([A-Z][\w.'-]+(?:\s*(?:,|and|&)\s*[A-Z][\w.'-]+)*)", a["task"])
        for grp in with_names:
            for n in _split_names(grp):
                attendees.add(n)
        for pre in ("to","for","by"):
            for n in re.findall(rf"\b{pre}\s+([A-Z][\w.'-]+)\b", a["task"]):
                attendees.add(n)
    for n in _find_names_anywhere(full_text):
        attendees.add(n)

    attendees_list = sorted(attendees)

    normalized_actions = []
    for a in actions:
        if a["assignees"]:
            for n in a["assignees"]:
                normalized_actions.append({"assignee": n, "task": a["task"], "raw": a["raw"], "section": "Next/Action"})
        else:
            normalized_actions.append({"assignee": None, "task": a["task"], "raw": a["raw"], "section": "Next/Action"})

    return {
        "file": filename,
        "time": time,
        "date": meeting_date_iso,
        "attendees": attendees_list,
        "actions": normalized_actions
    }

def build_index() -> Dict:
    meetings = []
    if not PARSED_DIR.exists():
        raise FileNotFoundError(f"{PARSED_DIR.resolve()} not found")

    for fp in sorted(PARSED_DIR.glob("*.txt")):
        text = fp.read_text(encoding="utf-8", errors="ignore")
        if fp.name.startswith("_"):
            continue
        meetings.append(parse_meeting_text(text, fp.name))

    people: Dict[str, List[Dict]] = {}
    for mt in meetings:
        for act in mt["actions"]:
            if not act["assignee"]:
                continue
            rec = {"file": mt["file"], "date": mt["date"], "time": mt["time"], "task": act["task"], "section": act["section"]}
            people.setdefault(act["assignee"], []).append(rec)

    index = {"meetings": meetings, "people_index": people}
    INDEX_JSON.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    # Optional CSV for quick audit
    try:
        import csv
        with INDEX_CSV.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["Assignee","Task","Date","Time","File"])
            for name, rows in people.items():
                for r in rows:
                    w.writerow([name, r["task"], r.get("date") or "", r.get("time") or "", r.get("file") or ""])
    except Exception:
        pass

    return index

def load_index() -> Dict:
    if INDEX_JSON.exists():
        return json.loads(INDEX_JSON.read_text(encoding="utf-8"))
    return build_index()

def find_tasks_by_person(name: str, date: Optional[str] = None) -> List[Dict]:
    idx = load_index()
    rows = idx.get("people_index", {}).get(name, [])
    if date:
        rows = [r for r in rows if r.get("date") == date]
    rows.sort(key=lambda r: (r.get("date") or "", r.get("time") or ""), reverse=True)
    return rows

def list_attendees(date: Optional[str] = None) -> List[str]:
    idx = load_index()
    attendees = set()
    for mt in idx.get("meetings", []):
        if date and mt.get("date") != date:
            continue
        for n in mt.get("attendees", []):
            attendees.add(n)
    return sorted(attendees)

def find_meeting_dates() -> List[str]:
    idx = load_index()
    return sorted({mt.get("date") for mt in idx.get("meetings", []) if mt.get("date")})
