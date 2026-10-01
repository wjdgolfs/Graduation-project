# 준비물 체크리스트 (데이터 · 가중치 · 패키지)

이 문서는 "무엇을 더 받아야 하고 무엇을 더 깔아야 하는가"만 다룹니다.
실험 설계와 진행 순서는 연구 계획 쪽을 따르세요.

측정 기준 환경 (2026-09-06 확인):

| 항목 | 값 | 영향 |
| --- | --- | --- |
| Python | 3.11.9 (`.venv`) | mediapipe 설치 가능 |
| PyTorch | 2.12.0+cu130 / torchvision 0.27.0 | 재설치 불필요 |
| GPU | RTX 5070 Laptop, **VRAM 8GB** | 배치 크기 제약. AMP 필수 |
| RAM | 31GB | 여유 |
| D: 여유 공간 | **137.6GB** | 데이터 계획의 실질적 상한 |
| ffmpeg | 설치됨 (winget) | xfade 전환 합성 가능 |

---

## 0. 지금 당장 할 일 — 승인에 시간이 걸리는 것부터

FF++ 는 구글 폼 승인까지 **약 1주**가 걸립니다. 폼부터 넣고 그동안 코드를 짜세요.

| 순서 | 항목 | 링크 | 대기 |
| --- | --- | --- | --- |
| 1 | FaceForensics++ 접근 신청 | <https://github.com/ondyari/FaceForensics#access> | **완료** |
| 2 | Celeb-DF v2 접근 신청 | <https://github.com/yuezunli/celeb-deepfakeforensics> | **완료** |

승인 대기 중에 할 수 있는 일: 패키지 설치, 부위/배경 마스크 추출 코드, ffmpeg 전환 합성 코드,
CBAM·ConvLSTM 구현, 평가 지표(고정 FPR 탐지율) 코드.

---

## 1. 다운로드할 데이터

FF++ 승인 메일로 받은 공식 스크립트를 `scripts/download_faceforensics.py` 로 두고 씁니다.
영상과 마스크는 이를 병렬로 감싼 `scripts/download_ffpp_batch.py` 로 받습니다.

### 1-1. 필수

```bash
# 주 데이터: 원본 + 4개 조작 기법 + 마스크, c23 (약 16GB). 연결 3개로 병렬 다운로드
# 공식 스크립트의 -d all 은 DeepFakeDetection(25GB)과 FaceShifter 까지 받으므로 쓰지 않습니다.
python scripts/download_ffpp_batch.py --workers 3
```

`raw` 는 500GB, 추출 PNG 는 2TB 입니다. **절대 받지 마세요.** c23 로 충분합니다.

#### 서버와 속도 (2026-09-06 실측)

| 서버 | 주소 | 상태 |
| --- | --- | --- |
| EU | `canis.vc.in.tum.de:8100` | **연결 안 됨** |
| CA | `falas.cmpt.sfu.ca:8100` | **연결 안 됨** |
| EU2 | `kaldir.vc.in.tum.de` (https) | 동작. 사실상 유일한 선택지 |

`--server` 를 바꿔서 속도를 개선할 수는 없습니다. 대신 **동시 연결 수**가 결정적입니다.

| 동시 연결 | 처리량 |
| --- | --- |
| 1개 | 약 24 KB/s |
| 4개 | 약 161 KB/s (6.7배) |
| 8개 | 서버가 응답을 끊음 |

공식 스크립트는 연결 하나로 순차 다운로드라 1000개에 40시간 넘게 걸립니다.
[`scripts/download_ffpp_batch.py`](../scripts/download_ffpp_batch.py) 가 같은 서버·같은 폴더 구조로
3~4개 연결을 써서 받습니다. 두 스크립트를 번갈아 써도 이어받기가 됩니다.

```bash
python scripts/download_ffpp_batch.py --workers 3
```

동시 연결을 4개 넘게 올리지 마세요. 학교 서버이고, 8개에서 차단이 걸렸습니다.
차단당하면 15분쯤 쉬었다 다시 실행하면 됩니다.

### 1-2. 조작 영역 마스크 — 부위별 기여도 분석의 정답지

```bash
python scripts/download_faceforensics.py D:/Graduation-project-data/raw/FaceForensics++ -d Deepfakes      -t masks --server EU2
python scripts/download_faceforensics.py D:/Graduation-project-data/raw/FaceForensics++ -d Face2Face      -t masks --server EU2
python scripts/download_faceforensics.py D:/Graduation-project-data/raw/FaceForensics++ -d FaceSwap       -t masks --server EU2
python scripts/download_faceforensics.py D:/Graduation-project-data/raw/FaceForensics++ -d NeuralTextures -t masks --server EU2
```

"어느 부위의 잔차가 튀는가"를 주장하려면 실제 조작 픽셀 위치와 대조해야 합니다.
마스크가 없으면 그 주장은 정성적 관찰에 머뭅니다.

### 1-3. 원본 유튜브 영상 38.5GB — **실제 점진적 전환 검증셋**

```bash
python scripts/download_faceforensics.py D:/Graduation-project-data/raw/FaceForensics++ -d original_youtube_videos --server EU2
```

FF++ 의 1000개 시퀀스는 유튜브 원본에서 **얼굴이 잘 추적되는 단일 샷 구간만 잘라낸 것**입니다.
잘려나간 나머지 구간에는 실제 편집자가 넣은 디졸브·페이드가 그대로 남아 있습니다.

이게 왜 중요하냐면, "인위적으로 합성한 전환이 현실적인가"라는 반박에 대해
**FF++ 와 완전히 같은 도메인의 실제 전환**으로 답할 수 있기 때문입니다.
같이 들어있는 프레임 정보 JSON 과 `conversion_dict.json` 으로
어느 구간이 데이터셋에 쓰였는지도 알 수 있으므로, 실제 전환 구간을 라벨링하기 쉽습니다.

38.5GB 는 이 목적 하나만으로도 값어치를 합니다.

서버가 구간 요청(`Accept-Ranges: bytes`)을 지원하므로(2026-09-12 확인, 정확히 38,549,639,418 바이트)
zip 끝의 파일 목록만 읽고 **필요한 영상만 골라 받을 수 있습니다.** 연결 하나로 전체를 받으면 2~3주가 걸리니 이 방식을 씁니다.
어떤 영상을 고를지는 함께 받는 정보 zip(`original_youtube_videos_info`, 6.8MB)의 추출 구간 정보로 정합니다.

### 1-4. 교차 데이터셋

| 데이터셋 | 용도 | 크기 | 비고 |
| --- | --- | --- | --- |
| Celeb-DF v2 | 교차 데이터셋 평가 | zip 9.27GB, 풀면 9.46GB | **받음 (2026-09-12)**. 590 + 5639 + 300 + 목록 txt 로 공식 개수와 일치. zip 안에 최상위 폴더가 없으니 풀 때 폴더를 지정 |
| DFD (DeepFakeDetection) | 여유 시 | c23 원본 3GB + 조작 22GB | FF++ 폼에 포함. 배우 28명 × 장면 16개 |

**DFDC 는 계획에서 뺐습니다 (2026-09-12).** Kaggle 대회가 2020년에 끝나 새 참가가 막혀 있을 수 있고,
Meta 공식 경로(dfdc.ai)는 AWS 계정이 필요합니다. 무엇보다 배우를 한 번에 찍은 10초 영상이라
장면 전환이 없어서 이 연구의 가설 검증에 쓸모가 적습니다.
"합성 전환만 썼다"는 지적은 1-3 의 FF++ 유튜브 원본(실제 전환)으로, 교차 데이터셋 평가는 Celeb-DF v2 로 대신합니다.

### 1-5. WildDeepfake — 계획에서 빼세요

WildDeepfake 는 **얼굴 크롭 시퀀스로만** 배포됩니다 (`deepfake_in_the_wild/*/N.tar.gz` 안이
전부 잘린 얼굴 이미지 폴더). **배경 픽셀이 존재하지 않으므로 이 연구의 배경 정규화를
적용할 수 없습니다.** 실제 전환 검증은 1-3 의 FF++ 유튜브 원본으로 합니다.

같은 이유로 DeepfakeBench 가 배포하는 전처리본(영상당 32프레임 얼굴 크롭)도
**주 데이터로 쓸 수 없습니다.** 가중치와 평가 프로토콜만 참고하고,
전처리는 원본 영상에서 직접 하세요.

### 1-6. 디스크 예산

| 항목 | 크기 |
| --- | --- |
| FF++ c23 videos | 10GB |
| FF++ masks | ~28GB (아래 실측 참고) |
| 유튜브 원본 | 38.5GB |
| Celeb-DF v2 (압축 해제) | 9.5GB |
| **소계** | **~63GB** |
| 여유 공간 (2026-09-12) | 137.2GB |
| 남는 공간 | **~74GB** |

남는 74GB 안에 전환 합성본과 전처리 캐시가 전부 들어가야 합니다.
전환 합성은 Cut/Fade/Dissolve × 길이 조합이라 원본의 여러 배로 불어납니다.

**실측 (2026-09-13)**
- Celeb-DF 얼굴 전처리 결과(.npz: 30프레임 크롭 + 랜드마크 + 배경 통계)는 영상 6,529개에 20.5GiB, 영상당 약 3.3MB 입니다.
- FF++ 5,000개(원본 1,000 + 조작 4,000)도 같은 방식이면 약 16GB 가 더 들 것으로 예상합니다.
- 원 논문 재현 체크포인트는 에폭당 약 500MB(모델 + Adam 상태)입니다.
- D: 여유 공간은 189GiB 입니다(Celeb-DF 전처리와 FF++ 일부 4.3GB 를 받은 뒤).

**실측 (2026-09-20)**
- FF++ c23 영상은 5종(원본 + 조작 4종) 1,000개씩 모두 받았고 합계 8.4GiB 입니다.
- 마스크는 기법마다 크기가 크게 다릅니다. Deepfakes 1,000개는 0.04GiB(파일당 0.04MB)인데,
  Face2Face 는 파일당 약 13.6MB 라 1,000개면 약 13.6GB 입니다. FaceSwap·NeuralTextures 마스크는 아직 안 받았습니다.
- 마스크 다운로드 중 서버가 응답을 끊어(`IncompleteRead`) 스크립트가 멈춘 적이 있습니다. 재시도가 이 오류도 잡도록 고쳤습니다.

- 프레임을 JPG/PNG 로 전부 펼치지 마세요. 필요한 클립만 그때그때 디코딩합니다.
- 전처리 결과(랜드마크, 부위 마스크, 배경 통계)는 이미지가 아니라
  영상당 하나의 `.npz` 로 저장하세요. 용량이 두 자릿수 배 차이 납니다.
- 전환 합성본은 mp4 로 저장하고, 합성 후 **원본과 같은 c23 로 재인코딩**하세요.
  전환 구간만 압축 이력이 다르면 모델이 전환이 아니라 코덱 흔적을 볼 수 있습니다.

---

## 2. 다운로드할 모델 가중치

pip 로 안 받아지는 것들입니다. `D:/Graduation-project-data/checkpoints/` 아래에 모으세요.

| 대상 | 받는 곳 | 크기 | 용도 |
| --- | --- | --- | --- |
| Xception (ImageNet) | `python scripts/setup_xception.py` → `checkpoints/xception/xception-43020ad28.pth` (**받음 2026-09-13**, sha256 `43020ad2…21aa90`) | ~88MB | 원 논문 백본. timm `legacy_xception` 의 기본 가중치(Keras ImageNet 가중치 변환본). timm 이 알아서 받으면 사용자 캐시 폴더로 가서 따로 받습니다 |
| **FTCN** | <https://github.com/yinglinzheng/FTCN/releases/download/weights/ftcn_tt.pth> | 59.2MB | **0단계 파일럿의 탐지기** + 6단계 일반성 검증. 바로 쓸 수 있는 공개 사전학습 시간 기반 탐지기는 이것뿐입니다. 공식 코드는 PyTorch 1.4 / Python 3.7 기준(pillow 6.1, scipy 1.5.3, pickle5 고정)이라 현재 환경으로 옮겨야 합니다 |
| FTCN 얼굴 검출 | <https://github.com/yinglinzheng/face_weights/releases/download/v1/mobilenet0.25_Final.pth> | 1.79MB | RetinaFace mobilenet0.25. FTCN 추론 전처리 |
| FTCN 얼굴 랜드마크 | <https://github.com/yinglinzheng/face_weights/releases/download/v1/mobilenet_224_model_best_gdconv_external.pth> | 15.24MB | FTCN 추론 전처리. 코드가 처음 실행될 때 자동으로 받습니다. FTCN 은 이 검출 → 랜드마크 → 정렬 크롭으로 학습됐으므로 파일럿에서도 같은 전처리를 써야 점수가 의미를 가집니다 |
| TransNet V2 | `pip install transnetv2-pytorch` | 휠 32.7MB | 샷 경계 검출 (4단계). 휠 크기가 가중치 파일(30.5MB)과 맞아 가중치가 함께 들어 있습니다. 공식 저장소 `inference-pytorch/` 에는 코드만 있고 가중치는 TensorFlow 로 직접 변환해야 합니다 |
| BiSeNet 얼굴 파싱 (선택) | HF `vivym/face-parsing-bisenet` → `79999_iter.pth` | 53MB | MediaPipe 대신 픽셀 단위 부위 분할 |
| DeepfakeBench (선택) | <https://github.com/SCLBD/DeepfakeBench> | — | 프레임 단위 탐지기 13종 가중치 + 통일된 평가 코드. **비디오 탐지기 가중치는 없음** (아래 참고) |

- CBAM 과 Bidirectional ConvLSTM 은 받을 게 없습니다. [`models/paper_bclstm.py`](../models/paper_bclstm.py) 에 직접 구현했습니다.
- **정정 (2026-09-12)**: 이전에 "AltFreezing / TALL 은 DeepfakeBench 에 가중치가 있다"고 적었는데 틀렸습니다.
  DeepfakeBench 릴리스(v1.0.0~v1.0.3)의 탐지기 가중치는 Xception·EfficientNet-B4·F3Net·SPSL 같은 프레임 단위 탐지기뿐이고,
  FTCN·TALL·AltFreezing 은 학습용 3D R50 백본(`I3D_8x8_R50.pth`, 112.5MB)만 있습니다. 탐지기로 쓰려면 FF++ 로 직접 학습해야 합니다.
- `transnetv2-pytorch` 는 원저자가 아닌 제3자(PyPI `allenday`)의 재구현입니다. 논문에 쓰기 전에 몇 개 영상에서 공식 결과와 일치하는지 확인하세요.
- DeepfakeBench 는 PyTorch 1.12, FTCN 공식 코드는 PyTorch 1.4 기준입니다. RTX 5070(sm_120)은 PyTorch 2.7 이상에서만 GPU 로 돌아가므로
  (현재 환경의 torch 2.12 는 sm_120 지원 확인) 어느 쪽이든 현재 환경으로 옮겨서 써야 합니다.

---

## 3. 설치할 패키지

```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

OpenCV 는 이미 설치된 기본판 `opencv-python` 을 그대로 씁니다. `opencv-python` / `opencv-contrib-python` / `opencv-python-headless` 는
모두 같은 `cv2` 를 설치해서 두 개 이상 있으면 import 가 깨지므로 배포판은 하나만 둡니다.

추가되는 것: `scenedetect`, `av`, `einops`, `seaborn`, `pytest`, 그리고 FTCN 추론용 `filterpy`, `fvcore`.

**정정 (2026-09-12)**: 처음 안내한 "opencv-python 삭제 → requirements 설치" 순서는 틀렸습니다.
설치될 패키지들의 요구 조건을 확인해 보니 cv2 를 요구하는 쪽이 갈려 있었습니다.

| 패키지 | 요구하는 OpenCV | 처리 |
| --- | --- | --- |
| `scenedetect` | `opencv-python` | 유지 (기본판과 맞음) |
| `batch-face` (+ `sixdrepnet`) | `opencv-python` | 뺌. 얼굴 검출은 FTCN 의 RetinaFace 로 충분 |
| `mediapipe` | `opencv-contrib-python` | **따로 설치** (아래). 부위 분할에 씀 |

**부위 분할은 MediaPipe Face Landmarker(478점)로 결정했습니다 (2026-09-13).**
mediapipe 본체는 `opencv-contrib-python` 을 요구해서 requirements.txt 에 넣지 않고, 의존성 자동 설치를 끄고 따로 설치합니다.
이 상태에서 기존 `opencv-python` 5.0 과 함께 import 되는 것을 확인했습니다.
`pip check` 에 나오는 opencv-contrib-python 경고는 예상된 것입니다.

```bash
.venv/Scripts/python.exe -m pip install --no-deps mediapipe==1.0.1
```

- mediapipe 1.0.1 에는 예전 `solutions.face_mesh` 가 없어서 Tasks API 의 `FaceLandmarker` 를 씁니다.
- 모델 파일: `D:/Graduation-project-data/checkpoints/mediapipe/face_landmarker.task` (3,758,596 바이트, sha256 `64184e22…`)
- FF++ 영상 035 앞 32프레임에서 모든 프레임의 얼굴을 찾았고, 프레임당 약 16.5ms(CPU)였습니다.
- Celeb-DF 는 얼굴 폭이 화면의 10% 안팎이라 전체 프레임을 넣으면 놓치는 영상이 있습니다. 그래서 얼굴 주변을 잘라 넣는 두 단계 검출을 씁니다
  ([`preprocessing/extract_faces.py`](../preprocessing/extract_faces.py) 머리말). Celeb-DF 6,529개 전처리(2026-09-13)에서 얼굴 찾은 프레임 비율은 99.99%,
  실패는 1개(`Celeb-real/id27_0005`, 프레임이 1장뿐인 영상)였습니다.
자세한 이유와 일부러 제외한 패키지는 [`requirements.txt`](../requirements.txt) 주석에 적었습니다.

---

## 4. 계획에 없지만 먼저 확인해야 하는 것

### 4-1. 배경이 실제로 존재하는가

FF++ 에는 얼굴이 화면 대부분을 차지하는 클로즈업이 적지 않습니다.
얼굴 박스를 확장해서 제외하고 나면 배경 픽셀이 거의 남지 않는 영상이 있고,
그런 영상에서 `D_bg` 는 노이즈가 됩니다. 잔차 `R_r = D_region − α·D_bg` 가
아예 성립하지 않는 구간이 생깁니다.

→ 전처리 단계에서 영상마다 **배경 유효 면적 비율**을 같이 저장하고,
   비율이 낮은 영상을 어떻게 처리할지(제외 / α 조정 / fallback)를
   실험 전에 정해두세요. 나중에 발견하면 결과 해석이 흔들립니다.

### 4-2. 8GB VRAM 안에 들어가는가 — **확인 (2026-09-13)**

원 논문 조건(배치 2 = real 1 + fake 1, 30프레임, 240×240)을 줄이지 않고 그대로 학습할 수 있습니다. bf16 AMP 기준 실측:

| 설정 | 스텝당 | 1 에폭 (5,539스텝) | 예약 VRAM |
| --- | --- | --- | --- |
| gradient checkpointing 켬 (기본) | 1.1초 | 약 102분 | 3.3GiB |
| gradient checkpointing 끔 | 0.83초 | 약 77분 | 6.25GiB |

- **`torch.backends.cudnn.benchmark` 는 끄세요.** 켰을 때 첫 스텝이 25~130초로 늘어졌고, 이어서 그래픽 드라이버 시간 초과 블루스크린
  (`VIDEO_TDR_FAILURE 0x116`)으로 PC 가 재부팅됐습니다. 끈 뒤에는 첫 스텝이 3초 안쪽입니다.
- 블루스크린 기록에 "자원 부족"도 함께 남아서, VRAM 여유가 큰 checkpointing 켬을 기본으로 둡니다.
- GPU 학습 중에는 CPU 전처리(프로세스 여러 개)를 같이 돌리지 마세요. 블루스크린이 났을 때 둘이 함께 돌고 있었습니다.

### 4-3. 원 논문 특정 — **확정**

이대현·문종섭, "Bidirectional Convolutional LSTM을 이용한 Deepfake 탐지 방법", 정보보호학회논문지 30(6), 2020.12, pp.1053–1065.
DOI 10.13089/JKIISC.2020.30.6.1053

| 항목 | 논문 값 | 이 코드에서 |
| --- | --- | --- |
| 데이터 | Celeb-DF: 학습 real 490 / fake 5,539, 테스트 100 / 100 | 어떤 영상인지 논문에 없어서 공식 테스트 목록에서 seed 0 으로 100 / 100 추출. real 1개(`id27_0005`, 1프레임 영상)가 빠져 학습 real 489 |
| 입력 | 영상당 30프레임, 240×240 RGB 얼굴 | 0번 프레임부터 연속 30프레임 (연속인지 균등 간격인지 논문에 없음) |
| 모델 | Xception → CBAM(공간 7×7) → Bi-ConvLSTM → FC 2 | CBAM 축소율 16, ConvLSTM 채널 128·필터 3×3 은 논문에 없어서 정한 값 |
| 학습 | lr 1e-4, 배치 2(real 1 + fake 1), Adam(0.9, 0.999), 교차 엔트로피 | Table 4 의 반복이 fake 를 한 번씩 다 쓰면 끝나므로 1 에폭 |
| 결과 | 정확도 93.5%, 정밀도 98.9%, 재현율 88.06%, F1 93.1% | 재현 목표는 정확도. 표 5 의 AUC 98.9 는 정밀도와 같은 값이라 오기로 의심 |
