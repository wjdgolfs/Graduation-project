# 점진적 장면 전환에 강건한 딥페이크 탐지

배경 기준 정규화로 전역(장면 전환)과 국소(합성 흔적) 시간 불연속을 분리하는 졸업 프로젝트입니다.

**가설**: 딥페이크 탐지기가 보는 "프레임 사이의 어색한 변화"에는 두 가지가 섞여 있습니다.
장면 전환처럼 화면 전체가 바뀌는 **전역** 변화와, 합성된 얼굴에서만 생기는 **국소** 변화입니다.
배경 변화량으로 부위 변화량을 정규화하면 둘을 분리할 수 있고, 장면 전환이 있는 영상에서 오탐이 줄어듭니다.

지금까지 측정으로 확인한 것:

- 2.5초 디졸브를 넣으면 FTCN 이 **진짜 영상 35개 중 32개를 가짜로 판정**합니다(전환이 없을 때는 16개).
- 실무 표준 샷 검출기는 그 디졸브를 **기본 설정에서 0%** 잡습니다.
- 배경 정규화는 전환이 만든 증가분의 **84~99%** 를 지우면서 real/fake 구분력은 유지했습니다.

수치와 근거는 [docs/02_실험결과.md](docs/02_실험결과.md)에 모두 있습니다.

## 진행 상태

| 단계 | 내용 | 상태 |
| --- | --- | --- |
| 0 | 파일럿: 전환이 시간 기반 탐지기를 흔드는지 확인 | 완료 |
| 1 | 원 논문 재현 (Xception + CBAM + Bi-ConvLSTM, Celeb-DF) | 완료 |
| 2 | 부위·배경 마스크 파이프라인, D_region / D_bg 계산 | 완료 |
| 3 | 전환 벤치마크 (Cut / Fade / Dissolve), 정규화와 FTCN 점수 검증 | 완료 |
| 4 | 샷 분할 기준선 (PySceneDetect) | 완료 |
| 5 | 배경 기준 정규화 모델 | 진행 중 |
| 6 | FTCN 일반성 검증 (원 논문 모델로도 채점) | 예비 결과 |
| 7 | 교차 데이터셋 평가 (FF++ → Celeb-DF) | 예정 |

## 문서

| 문서 | 내용 |
| --- | --- |
| [docs/00_setup_checklist.md](docs/00_setup_checklist.md) | 받아야 할 데이터와 가중치, 설치할 패키지, 환경에서 확인한 제약 |
| [docs/01_진행현황.md](docs/01_진행현황.md) | 단계별 상태, 주요 결과 해설, 코드 지도, 겪은 문제와 대처 |
| [docs/02_실험결과.md](docs/02_실험결과.md) | 실험 수치 모음과 재현 명령 |
| [docs/05_논문자료.md](docs/05_논문자료.md) | **논문 작성용 자료**: 논문 구성, 핵심 주장과 근거, 실험 설정, 결과 표, 그림 목록, 예상 질문 |
| [docs/04_5단계_설계.md](docs/04_5단계_설계.md) | 제안 모델 설계안: 구조, 비교할 기준선, 평가 프로토콜, 구현 순서 |
| [docs/03_초기_장면탐지_실험.md](docs/03_초기_장면탐지_실험.md) | 초기 프로토타입(밝기 차이 기반 장면 탐지)과 기본 용어 설명. 예전 README |

## 저장소 구조

```
preprocessing/   영상 → 얼굴 크롭·랜드마크·배경 통계, 부위 신호, 전환 벤치마크 생성
models/          원 논문 재현 모델(Xception + CBAM + Bi-ConvLSTM), 제안 모델, FTCN 실행 래퍼
scripts/         학습·분석·다운로드·가중치 준비 스크립트
tests/           단위 테스트 63개 (전환 계산, 부위 마스크, 모델 구조, 다운로드 재시도)
configs/         모든 경로와 기준값이 모인 config.yaml
results/         실험 결과 CSV (용량이 작은 것만 저장소에 포함)
docs/            위 문서들
```

## 실행 순서

```bash
# 0) 준비: 가중치 내려받기 (한 번만)
python scripts/setup_mediapipe.py
python scripts/setup_xception.py
python scripts/setup_ftcn.py

# 1) 얼굴 전처리: 영상 → 30프레임 얼굴 크롭 + 랜드마크 + 배경 통계
python preprocessing/extract_faces.py --dataset celebdf
python preprocessing/extract_faces.py --dataset ffpp

# 2) 부위 신호: 눈·입·얼굴 경계·얼굴 전체의 변화량과 배경 변화량
python preprocessing/region_signals.py --dataset ffpp
python scripts/summarize_regions.py --dataset ffpp

# 3) 전환 벤치마크: 전환 9종을 합성한 클립과 그 신호
python preprocessing/build_transition_benchmark.py
python preprocessing/benchmark_signals.py
python scripts/analyze_benchmark.py

# 3-1) 전환이 탐지 점수를 얼마나 흔드는지 (FTCN 채점 210클립, 약 2시간)
python models/ftcn_runner.py \
    --manifest D:/Graduation-project-data/processed/benchmark/manifest_pass1.csv \
    --out-dir D:/Graduation-project-data/processed/benchmark/ftcn
python scripts/analyze_benchmark_ftcn.py
python scripts/compare_ftcn_signals.py

# 3-2) 같은 클립을 원 논문 모델로도 채점 (6단계, 전처리 1분 + 채점 2분)
python preprocessing/benchmark_faces.py --manifest D:/Graduation-project-data/processed/benchmark/manifest_pass1.csv
python scripts/score_benchmark_paper.py --checkpoint D:/Graduation-project-data/checkpoints/paper_bclstm/paper_repro/epoch_01.pt

# 4) 샷 분할 기준선
python scripts/detect_shots_benchmark.py

# 원 논문 재현 학습 (GPU, 1 에폭 약 100분)
python scripts/train_paper.py
```

## 데이터

영상과 전처리 결과는 용량이 커서 저장소에 없습니다(`D:/Graduation-project-data`, 약 90GB).
받는 방법은 [docs/00_setup_checklist.md](docs/00_setup_checklist.md)에 적어 두었습니다.

| 항목 | 규모 |
| --- | --- |
| FaceForensics++ 영상 (원본 + 조작 4종) | 5,000개 |
| Celeb-DF v2 영상 | 6,529개 |
| 얼굴 전처리 결과 | 11,528개 · 40GB |
| 전환 벤치마크 클립 | 1,260개 · 883MB |
| 벤치마크 FTCN 채점 | 210클립 |

## 환경에서 확인한 제약

- GPU 학습 전에 `torch.backends.cudnn.benchmark` 를 **끄세요.** 켠 상태로 돌리다가 그래픽 드라이버 시간 초과
  블루스크린(`VIDEO_TDR_FAILURE 0x116`)으로 PC 가 재부팅된 적이 있습니다.
- GPU 작업 중에 CPU 전처리를 함께 돌리지 마세요.
- 8GB VRAM 기준, 원 논문 모델은 gradient checkpointing 을 켜면 3.3GB 로 돕니다(스텝당 1.1초).
