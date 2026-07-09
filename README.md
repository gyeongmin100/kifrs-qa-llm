# KIFRS-QA-LLM: 회계기준 질의회신 특화 LLM (LoRA + RAG)

한국회계기준원·금융감독원의 **회계기준 질의회신 사례**를 활용해, 오픈소스 LLM에 도메인 지식을 주입하는 두 가지 접근을 구현하고 비교한 프로젝트입니다.

- **LoRA 파인튜닝** — 회계기준·문단을 인용하며 결론을 제시하는 회계 자문 답변 스타일 학습
- **RAG** — 과거 유사 질의회신 사례를 검색해 근거로 제공
- **비교 평가** — Base / RAG / LoRA / LoRA+RAG 4개 구성을 동일한 홀드아웃 문항으로 평가

회계 도메인은 데이터 민감성이 높아 외부 API 대신 **로컬(온프레미스) LLM**이 요구되는 경우가 많습니다. 이 프로젝트는 오픈소스 모델(Qwen2.5-7B-Instruct)만으로 도메인 특화 어시스턴트를 구축하는 전 과정을 다룹니다.

## 아키텍처

```
KIFRS.com 질의회신 (2,255건)
        │  scrape.py
        ▼
원본 JSONL ── prepare_data.py ──► train.jsonl (2,114건)
                                  test.jsonl  (30건, 홀드아웃)
                                  rag_corpus.jsonl (train과 동일, test 제외)
        │
        ├─► [LoRA] Qwen2.5-7B-Instruct + bf16 LoRA (TRL SFTTrainer)
        └─► [RAG]  bge-m3 임베딩 + FAISS 유사 사례 top-3 검색
        │
        ▼
평가: 30문항 × 4구성, 정답(실제 회신) 대비 임베딩 유사도 + 정성 비교
```

## 실행 방법

### 1. 데이터 수집 (로컬)

```bash
pip install -r requirements.txt
python src/scrape.py --rows 200 --sleep 0.3   # 전체 약 2,255건 수집
python src/prepare_data.py                     # 질의/회신 파싱, train/test 분리
```

### 2. 학습 + 평가 (Colab)

`notebooks/train_and_eval_colab.ipynb`를 Colab(A100 권장)에서 열고 순서대로 실행합니다. 노트북 안에서 저장소 클론 → 데이터 수집 → LoRA 학습 → RAG 인덱스 → 4구성 비교 평가까지 진행됩니다.

## 평가 설계

- **홀드아웃**: 테스트 30문항은 학습 데이터와 RAG 인덱스 양쪽에서 제외 (데이터 누출 방지)
- **자동 지표**: 모델 답변과 실제 회신(정답)의 bge-m3 임베딩 코사인 유사도
- **정성 평가**: 전체 답변을 `results/eval_answers.json`에 저장해 결론 일치 여부·기준서 인용 정확성·환각 여부를 수동 확인

## 데이터 파싱

원문은 출처(회계기준원 정규질의/신속처리질의, 금감원, IFRS 해석위원회)마다 섹션 표기가 달라(`질의/회신`, `질의 요약/회신 요약`, `Ⅲ. 회신`, `회신□...` 등) 정규식 기반 파서로 질문부/답변부를 분리했습니다. 전체 2,255건 중 2,144건(95.1%) 파싱에 성공했으며, 나머지는 회신 섹션이 없거나 비정형 구조인 문서입니다.

## 저장소 구조

```
├── notebooks/
│   ├── train_and_eval_colab.ipynb   # 학습·RAG·평가 통합 노트북 (Colab, 1회 실행)
│   └── demo_colab.ipynb             # 데모 전용: 저장된 어댑터 로드 + Gradio 채팅 (~10분)
├── src/
│   ├── scrape.py                    # KIFRS.com 질의회신 수집
│   └── prepare_data.py              # 파싱, train/test 분리
├── data/
│   └── sample/sample.jsonl          # 데모용 샘플 5건 (원본 데이터는 미포함)
├── results/                         # 평가 결과 (노트북 실행 시 생성)
└── requirements.txt
```

## 데이터 출처 및 고지

- 질의회신 원문 출처: [KIFRS.com 질의회신요약](https://www.kifrs.com/qnas/) (한국회계기준원·금융감독원 공개 자료)
- 본 저장소는 **학습·연구 목적**으로만 사용되며, 원본 데이터 전문은 저작권 고려로 포함하지 않습니다. 위 스크립트로 직접 수집해야 합니다.
- 신속처리질의는 한국회계기준원의 공식 의견이 아니며, 모델 출력은 실제 회계 자문을 대체할 수 없습니다.
