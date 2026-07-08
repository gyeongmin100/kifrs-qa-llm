"""
수집한 KIFRS 질의회신 원문을 학습/평가/RAG용 데이터로 변환한다.

- fullContent를 질문부(현황·배경·질의)와 답변부(회신·판단근거)로 분리
- instruction 형식(train.jsonl)과 홀드아웃 평가셋(test.jsonl)으로 분할
- RAG 인덱싱용 코퍼스(rag_corpus.jsonl)는 train 문서만 포함 (데이터 누출 방지)
"""
import argparse
import html
import json
import random
import re
from pathlib import Path

# 마커 앞에 붙을 수 있는 접두어: 마크다운 헤더(#), 번호(3. / Ⅲ. / II.)
_PREFIX = r"(?:#+\s*)?(?:(?:\d+|[ⅠⅡⅢⅣⅤⅥⅦ]|[IVX]{1,4})\s*[.)]?\s*)?"
# 답변부 시작 마커 (타입별 상이)
ANSWER_MARKERS = [
    r"회신\s*요약",
    r"검토 내용과 결정",
    r"회신\s*내용",
    r"회신",
]
# 답변부 이후 잘라낼 꼬리 섹션
TAIL_MARKERS = [
    rf"\n\s*{_PREFIX}참고자료\s*\n",
    rf"\n\s*{_PREFIX}관련 회계기준\s*\n",
    rf"\n\s*{_PREFIX}실무적용지침\s*\n",
]

TYPE_NAMES = {
    11: "회계기준원(K-IFRS)",
    12: "회계기준원(일반기업회계기준)",
    13: "IFRS해석위원회",
    14: "신속처리질의(일반기업회계기준)",
    15: "신속처리질의(K-IFRS)",
    24: "금융감독원(일반기업회계기준)",
    25: "금융감독원(K-IFRS)",
}


def clean(text: str) -> str:
    text = html.unescape(text)
    text = text.replace("\\n", "\n")  # 일부 원문에 이스케이프된 개행 문자열 존재
    text = text.replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_record(item: dict) -> dict | None:
    fc = item.get("fullContent") or ""
    # 머리말(관련 회계기준 요약줄) 제거: '본문' 이후만 사용
    body_idx = fc.find("본문")
    body = fc[body_idx + len("본문"):] if body_idx >= 0 else fc
    body = clean(body)

    # 답변부 시작 위치 탐색 (가장 먼저 나오는 마커 채택)
    # 마커 뒤는 줄바꿈 또는 항목기호(□) 허용 — '회신□...'처럼 붙는 원문 존재
    padded = "\n" + body
    ans_start, ans_end = None, None
    for pat in ANSWER_MARKERS:
        # 앞: 줄바꿈 또는 질문 끝('?회신□', ')회신□' 형태)
        # 뒤: 공백, 각주([4]), 괄호, 번호(회신1.), 항목기호(□◦ㅇｏ, 'o ') 허용
        m = re.search(rf"(?<=[\n?)])\s*{_PREFIX}{pat}(?:\[\d+\])?\s*(?=\s|□|◦|ㅇ|ｏ|\(|\d|o\s)", padded)
        if m and (ans_start is None or m.start() < ans_start):
            ans_start, ans_end = m.start(), m.end()
    if ans_start is None:
        return None

    question = padded[:ans_start].strip()
    answer = padded[ans_end:].strip()

    # 답변부 꼬리 섹션 절단
    tail_idx = len(answer)
    for pat in TAIL_MARKERS:
        m = re.search(pat, "\n" + answer)
        if m and m.start() < tail_idx:
            tail_idx = m.start()
    answer = answer[:tail_idx].strip()

    # 질문부 머리의 '질의' 계열 마커 제거 ('질의\n', '질의◦', '2. 질의 요약' 등)
    question = re.sub(rf"^{_PREFIX}(배경 및 질의|질의(\s*(요약|사항|내용))?)(?=\s|□|◦|ㅇ|ｏ)\s*", "", question).strip()

    # 답변은 '갑설이 타당함'처럼 짧아도 유효 (질문에 갑설/을설 정의 포함)
    if len(question) < 30 or len(answer) < 5:
        return None

    return {
        "docNumber": item["docNumber"],
        "title": item["title"],
        "type": item["type"],
        "source": TYPE_NAMES.get(item["type"], str(item["type"])),
        "date": item.get("date"),
        "question": question,
        "answer": answer,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw/kifrs_qna.jsonl")
    parser.add_argument("--outdir", default="data/processed")
    parser.add_argument("--test-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    items = [json.loads(l) for l in open(args.raw, encoding="utf-8")]
    parsed, skipped = [], 0
    for item in items:
        rec = parse_record(item)
        if rec:
            parsed.append(rec)
        else:
            skipped += 1
    print(f"파싱 성공 {len(parsed)}건 / 스킵 {skipped}건 (전체 {len(items)}건)")

    rng = random.Random(args.seed)
    rng.shuffle(parsed)
    test, train = parsed[: args.test_size], parsed[args.test_size:]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    def dump(path: Path, records: list[dict]):
        with path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{path}: {len(records)}건")

    dump(outdir / "train.jsonl", train)
    dump(outdir / "test.jsonl", test)
    # RAG 코퍼스: train 문서만 (test 유출 방지)
    dump(outdir / "rag_corpus.jsonl", train)

    # 저장소 공개용 샘플 (원문 출처 고지와 함께 소량만)
    sample_dir = Path("data/sample")
    sample_dir.mkdir(parents=True, exist_ok=True)
    dump(sample_dir / "sample.jsonl", train[:5])


if __name__ == "__main__":
    main()
