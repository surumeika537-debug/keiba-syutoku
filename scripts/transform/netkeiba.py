"""Parse a netkeiba race result page (https://db.netkeiba.com/race/<id>/) into structured rows.

Defensive: missing fields become None. Tolerates minor layout drift.
"""
from __future__ import annotations

import re
from datetime import date as date_cls

from bs4 import BeautifulSoup, Tag

from src.utils import (
    normalize_bet_type,
    normalize_finish_position,
    normalize_trifecta_combination,
    parse_float_safe,
    parse_int_safe,
    setup_logger,
)
from scripts.transform.base import BaseParser, ParsedRace

log = setup_logger("transform.netkeiba")

# netkeiba course codes (positions 5-6 of race_id). 01-10 are JRA; 30+ are NAR.
COURSE_CODE_TO_NAME = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
    "30": "門別", "35": "盛岡", "36": "水沢", "42": "浦和", "43": "船橋",
    "44": "大井", "45": "川崎", "46": "金沢", "47": "笠松", "48": "名古屋",
    "50": "園田", "51": "姫路", "54": "高知", "55": "佐賀",
}
JRA_COURSE_CODES = {f"{i:02d}" for i in range(1, 11)}

# Surface: 芝 / ダ / ダート / 障芝 / 障ダ. The chars between the surface marker and the
# distance are wildly inconsistent (e.g. "芝右 内2周3600m" where 2周=two laps, "芝直線1000m",
# "障芝 外-内3250m"). Use a non-greedy any-char skip — the distance is always the LAST
# numeric run before "m" on the conditions line, and `.` won't traverse newlines.
SURFACE_PAT = re.compile(r"(障?(?:芝|ダ(?:ート)?)).*?(\d{3,4})\s*m", re.UNICODE)
# Track condition: separate "芝 : 良" or "ダート : 良" or "馬場 : 良"
TRACK_COND_PAT = re.compile(r"(?:馬場|芝|ダート|ダ)\s*[:：]\s*(良|稍重|重|不良)")
DATE_PAT = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日")
# Grade markers found in the race-name H1: (G1)/(GI)/(JGI)/(JpnI) etc. Map to G1/G2/G3.
GRADE_TOKEN_PAT = re.compile(r"\(\s*(J?Pn?|J)?\s*G?(I{1,3}|[1-3])\s*\)", re.IGNORECASE)
_ROMAN_TO_INT = {"I": 1, "II": 2, "III": 3}


def _normalize_grade(token_full: str) -> str | None:
    """'(GI)' / '(G1)' / '(JpnIII)' / '(JGII)' -> 'G1'/'G2'/'G3'."""
    m = re.search(r"(I{1,3}|[1-3])\s*\)", token_full)
    if not m:
        return None
    num_token = m.group(1)
    n = _ROMAN_TO_INT.get(num_token.upper()) or int(num_token)
    return f"G{n}" if 1 <= n <= 3 else None


class NetkeibaParser(BaseParser):
    name = "netkeiba"

    def parse(self, html: str, race_id: str) -> ParsedRace:
        soup = BeautifulSoup(html, "lxml")
        race = self._parse_race(soup, race_id)
        entries = self._parse_entries(soup, race_id)
        race["field_size"] = race.get("field_size") or len(entries) or None
        payouts = self._parse_payouts(soup, race_id)
        return ParsedRace(race=race, entries=entries, payouts=payouts)

    # ---- race header -------------------------------------------------------

    def _parse_race(self, soup: BeautifulSoup, race_id: str) -> dict:
        # First <h1> is the netkeiba logo (empty text). The race-name h1 is the next non-empty one.
        race_name = None
        for h1 in soup.find_all("h1"):
            txt = h1.get_text(strip=True)
            if txt:
                race_name = txt
                break

        # conditions string e.g. "障芝4100m / 天候 : 晴 / 芝 : 良 / 発走 : 15:05"
        # On netkeiba db pages this lives inside <dl class="racedata"> or <diary_snap_cut>.
        condition_text = ""
        racedata = soup.find("dl", class_="racedata")
        if racedata:
            condition_text = racedata.get_text(" ", strip=True)
        else:
            snap = soup.find("diary_snap_cut") or soup.find(class_="diary_snap_cut")
            if snap:
                condition_text = snap.get_text(" ", strip=True)

        surface = distance = track_condition = None
        m = SURFACE_PAT.search(condition_text)
        if m:
            raw_surface = m.group(1)
            if raw_surface.startswith("障"):
                surface = "障害"
            elif raw_surface == "芝":
                surface = "芝"
            else:
                surface = "ダート"
            distance = parse_int_safe(m.group(2))
        m2 = TRACK_COND_PAT.search(condition_text)
        if m2:
            track_condition = m2.group(1)

        # date — from .smalltxt block ("YYYY年M月D日 ...")
        race_date = None
        smalltxt = soup.find(class_="smalltxt")
        if smalltxt:
            d = DATE_PAT.search(smalltxt.get_text(" ", strip=True))
            if d:
                try:
                    race_date = date_cls(int(d.group(1)), int(d.group(2)), int(d.group(3)))
                except ValueError:
                    race_date = None

        # grade — search the race name and the title-bar (racedata) for (G1)/(GI)/(JGII)/(JpnIII) etc.
        grade = None
        for src in (race_name or "", condition_text):
            m3 = GRADE_TOKEN_PAT.search(src)
            if m3:
                g = _normalize_grade(m3.group(0))
                if g:
                    grade = g
                    break

        racecourse = COURSE_CODE_TO_NAME.get(race_id[4:6])

        return {
            "race_id": race_id,
            "race_date": race_date,
            "race_name": race_name,
            "grade": grade,
            "racecourse": racecourse,
            "surface": surface,
            "distance": distance,
            "track_condition": track_condition,
            "field_size": None,  # filled in by caller from entries
        }

    # ---- result table ------------------------------------------------------

    def _find_result_table(self, soup: BeautifulSoup) -> Tag | None:
        # modern: <table class="race_table_01 nk_tb_common">
        for cls in ("race_table_01", "race_table_old", "RaceTable01"):
            t = soup.find("table", class_=cls)
            if t:
                return t
        # heuristic fallback: largest table on the page
        tables = soup.find_all("table")
        return max(tables, key=lambda t: len(t.find_all("tr")), default=None)

    def _parse_entries(self, soup: BeautifulSoup, race_id: str) -> list[dict]:
        table = self._find_result_table(soup)
        if not table:
            log.warning("race_id=%s: no result table", race_id)
            return []

        # locate columns by header text
        header_cells = []
        head = table.find("tr")
        if head:
            header_cells = [c.get_text(strip=True) for c in head.find_all(["th", "td"])]

        def col(name: str) -> int | None:
            for i, h in enumerate(header_cells):
                if name in h:
                    return i
            return None

        col_pos = col("着順")
        col_frame = col("枠")
        col_num = col("馬番")
        col_horse = col("馬名")
        col_jockey = col("騎手")
        col_pop = col("人気")
        col_odds = col("単勝")

        entries: list[dict] = []
        for tr in table.find_all("tr")[1:]:
            tds = tr.find_all(["td", "th"])
            if not tds:
                continue

            def cell(idx: int | None) -> str | None:
                if idx is None or idx >= len(tds):
                    return None
                return tds[idx].get_text(" ", strip=True)

            horse_num = parse_int_safe(cell(col_num))
            if horse_num is None:
                continue  # skip non-data rows

            fin_int, fin_status = normalize_finish_position(cell(col_pos))
            entries.append({
                "race_id": race_id,
                "horse_number": horse_num,
                "frame_number": parse_int_safe(cell(col_frame)),
                "horse_name": cell(col_horse),
                "jockey": cell(col_jockey),
                "popularity": parse_int_safe(cell(col_pop)),
                "win_odds": parse_float_safe(cell(col_odds)),  # REAL; "---" 等は None
                "finish_position": fin_int,                    # 完走時のみ
                "finish_status": fin_status,                   # 完走 / 中止 / etc.
            })
        return entries

    # ---- payouts -----------------------------------------------------------

    def _parse_payouts(self, soup: BeautifulSoup, race_id: str) -> list[dict]:
        rows: list[dict] = []
        for table in soup.find_all("table", class_=re.compile(r"pay_table_0[12]")):
            for tr in table.find_all("tr"):
                th = tr.find("th")
                tds = tr.find_all("td")
                if not th or len(tds) < 2:
                    continue
                bet_type = th.get_text(strip=True)
                # Preserve <br> as newline before splitting
                for br in tds[0].find_all("br"):
                    br.replace_with("\n")
                for br in tds[1].find_all("br"):
                    br.replace_with("\n")
                combos = [c.strip() for c in tds[0].get_text("\n").split("\n") if c.strip()]
                pays = [p.strip() for p in tds[1].get_text("\n").split("\n") if p.strip()]
                bet_type_canonical = normalize_bet_type(bet_type)
                for combo, pay in zip(combos, pays):
                    rows.append({
                        "race_id": race_id,
                        "bet_type": bet_type_canonical,
                        "combination": normalize_trifecta_combination(combo),
                        "payout_yen": parse_int_safe(pay),
                    })
        return rows
