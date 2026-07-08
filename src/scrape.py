"""
KIFRS.com 질의회신요약 스크래퍼.

https://www.kifrs.com/api/qnas/v2 를 페이지네이션하며 회계기준원/금융감독원
질의회신 전문(fullContent 포함)을 그대로 JSONL로 저장한다.
Q&A 학습데이터 변환 등 후처리는 별도 스크립트에서 담당한다.
"""
import argparse
import json
import time
from pathlib import Path

import requests

API_URL = "https://www.kifrs.com/api/qnas/v2"
HEADERS = {"User-Agent": "kifrs-qna-scraper-portfolio-project/1.0"}

FIELDS = ["id", "type", "docNumber", "date", "title", "fullContent", "relStds", "tags"]


def fetch_page(session: requests.Session, page: int, rows: int) -> list[dict]:
    resp = session.get(API_URL, headers=HEADERS, params={"types": "all", "page": page, "rows": rows}, timeout=30)
    resp.raise_for_status()
    return resp.json()["facilityQnas"]


def fetch_all(rows: int, sleep: float, limit: int | None) -> list[dict]:
    session = requests.Session()
    items = []
    page = 1
    while True:
        batch = fetch_page(session, page, rows)
        if not batch:
            break
        items.extend(batch)
        print(f"page {page}: {len(batch)}건 (누적 {len(items)}건)")
        if limit and len(items) >= limit:
            items = items[:limit]
            break
        if len(batch) < rows:
            break
        page += 1
        time.sleep(sleep)
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=200, help="페이지당 항목 수")
    parser.add_argument("--sleep", type=float, default=0.5, help="페이지 간 대기(초)")
    parser.add_argument("--limit", type=int, default=None, help="테스트용 최대 수집 건수")
    parser.add_argument("--out", type=str, default="data/raw/kifrs_qna.jsonl")
    args = parser.parse_args()

    items = fetch_all(args.rows, args.sleep, args.limit)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in items:
            record = {k: item.get(k) for k in FIELDS}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"총 {len(items)}건 저장 -> {out_path}")


if __name__ == "__main__":
    main()
