"""Utility to snapshot the ASOCA challenge leaderboard (top-N entries)."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import List

import requests
from bs4 import BeautifulSoup


LEADERBOARD_URL = "https://asoca.grand-challenge.org/evaluation/challenge/leaderboard/"


@dataclass
class LeaderboardEntry:
    """Structured representation of a leaderboard row."""

    rank: str
    user: str
    date: str
    dice: float
    hd95: float
    evaluation_url: str

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "user": self.user,
            "date": self.date,
            "dice": self.dice,
            "hd95": self.hd95,
            "evaluation_url": self.evaluation_url,
        }


def _post_payload(start: int, length: int) -> dict:
    payload = {
        "draw": 1,
        "columns[0][data]": 0,
        "columns[1][data]": 1,
        "columns[2][data]": 2,
        "columns[3][data]": 3,
        "columns[4][data]": 4,
        "order[0][column]": 0,
        "order[0][dir]": "asc",
        "start": start,
        "length": length,
        "search[value]": "",
        "search[regex]": "false",
    }
    return payload


def fetch_leaderboard(limit: int) -> List[LeaderboardEntry]:
    session = requests.Session()
    headers = {"User-Agent": "Mozilla/5.0"}

    # Prime CSRF token
    response = session.get(LEADERBOARD_URL, headers=headers, timeout=30)
    response.raise_for_status()

    token = session.cookies.get("_csrftoken")
    headers.update(
        {
            "Referer": LEADERBOARD_URL,
            "X-CSRFToken": token,
            "X-Requested-With": "XMLHttpRequest",
            "Accept": "application/json",
        }
    )

    entries: List[LeaderboardEntry] = []
    start = 0
    while len(entries) < limit:
        payload = _post_payload(start=start, length=min(10, limit - len(entries)))
        resp = session.post(LEADERBOARD_URL, headers=headers, data=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()["data"]
        for row in data:
            rank = BeautifulSoup(row[0], "html.parser").get_text(strip=True)
            user = BeautifulSoup(row[1], "html.parser").get_text(strip=True)
            date = BeautifulSoup(row[2], "html.parser").get_text(strip=True)
            dice_anchor = BeautifulSoup(row[3], "html.parser").find("a")
            hd_anchor = BeautifulSoup(row[4], "html.parser").find("a")
            dice = float(dice_anchor.get_text(strip=True))
            hd95 = float(hd_anchor.get_text(strip=True))
            evaluation_url = dice_anchor["href"]
            entries.append(
                LeaderboardEntry(
                    rank=rank,
                    user=user,
                    date=date,
                    dice=dice,
                    hd95=hd95,
                    evaluation_url=evaluation_url,
                )
            )
        start += len(data)
        if not data:
            break

    return sorted(entries[:limit], key=lambda e: e.dice, reverse=True)


def main(limit: int, output: str) -> None:
    entries = fetch_leaderboard(limit)
    with open(output, "w", encoding="utf-8") as fh:
        json.dump([entry.to_dict() for entry in entries], fh, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ASOCA leaderboard snapshot.")
    parser.add_argument("--limit", type=int, default=5, help="Number of entries to export.")
    parser.add_argument(
        "--output",
        type=str,
        default="data/asoca_leaderboard_top5.json",
        help="Destination JSON path.",
    )
    args = parser.parse_args()
    main(args.limit, args.output)
