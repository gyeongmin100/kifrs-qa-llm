"""
OpenAI API로 4구성(base/rag/lora/lora_rag) 답변의 결론 정오(O/X)를 채점한다.

임베딩 코사인 유사도는 '주제가 비슷한 오답'에 관대하다는 한계가 있어,
정답과 모델 답변의 최종 결론이 실제로 일치하는지를 LLM으로 직접 판정한다.

사용법:
    export OPENAI_API_KEY=sk-...
    python src/llm_judge.py --in results/eval_answers.json --out results/judge_scores_gpt-5.4.json
"""
import argparse
import json
import time
from pathlib import Path

from openai import OpenAI

CONFIGS = ["base", "rag", "lora", "lora_rag"]

JUDGE_SCHEMA = {
    "name": "judge_verdict",
    "schema": {
        "type": "object",
        "properties": {
            "match": {
                "type": "boolean",
                "description": "정답과 모델 답변의 최종 결론이 일치하면 true",
            },
            "reason": {
                "type": "string",
                "description": "판정 근거 한 문장",
            },
        },
        "required": ["match", "reason"],
        "additionalProperties": False,
    },
    "strict": True,
}

JUDGE_PROMPT = """당신은 회계기준 질의회신의 정답과 모델 답변을 비교하는 채점자입니다.
문체·근거 서술의 차이는 무시하고, 오직 '최종 결론'(허용/불허, 자산/부채 분류,
당기손익/기타포괄손익 여부 등)이 정답과 일치하는지만 판단하세요.
모델 답변이 반복되거나 장황해도 결론만 맞으면 일치로 판정합니다.
정답과 다른 결론을 내렸거나, 결론을 제시하지 못했거나, 사실과 다른 근거(환각)로
결론에 도달한 경우는 불일치로 판정합니다.

[질의]
{question}

[정답]
{reference}

[모델 답변]
{answer}
"""


def judge_one(client: OpenAI, model: str, question: str, reference: str, answer: str) -> dict:
    last_err = None
    for attempt in range(3):
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                question=question, reference=reference, answer=answer[:2000]
            )}],
            response_format={"type": "json_schema", "json_schema": JUDGE_SCHEMA},
            temperature=0,
        )
        content = resp.choices[0].message.content
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            last_err = e
            time.sleep(1)
    return {"match": None, "reason": f"judge_error: {last_err}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default="results/eval_answers.json")
    parser.add_argument("--out", dest="out_path", default="results/judge_scores_gpt-5.4.json")
    parser.add_argument("--model", default="gpt-5.4")
    args = parser.parse_args()

    records = json.loads(Path(args.in_path).read_text(encoding="utf-8"))
    client = OpenAI()
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for i, rec in enumerate(records):
        row = {"docNumber": rec["docNumber"]}
        for cfg in CONFIGS:
            verdict = judge_one(client, args.model, rec["question"], rec["reference"], rec[cfg])
            row[cfg] = verdict
            time.sleep(0.2)
        results.append(row)
        # 중간 저장: 도중에 실패해도 진행분은 보존
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{i + 1}/{len(records)}] {rec['docNumber']}: "
              + " ".join(f"{c}={'O' if row[c]['match'] else ('?' if row[c]['match'] is None else 'X')}"
                         for c in CONFIGS))

    print("\n=== 결론 정답률 (judge_error 제외) ===")
    for cfg in CONFIGS:
        valid = [r[cfg]["match"] for r in results if r[cfg]["match"] is not None]
        errors = len(results) - len(valid)
        acc = sum(valid) / len(valid) if valid else 0
        err_note = f", 채점실패 {errors}건" if errors else ""
        print(f"{cfg:10s}: {acc:.1%} ({sum(valid)}/{len(valid)}{err_note})")


if __name__ == "__main__":
    main()
