# Graduation-project

영상에서 장면이 갑자기 바뀌거나 서서히 전환되는 부분을 찾아보는 졸업 프로젝트 실험 코드입니다. 현재 구현은 테스트 영상 생성, 프레임 추출, 밝기 차이 기반 장면 전환 분석, 결과 시각화, 텐서 입력 형태 확인까지 포함합니다.

## 이 코드는 왜 필요한가요?

영상은 여러 장의 이미지가 시간순으로 이어진 자료입니다. 장면이 바뀌는 위치를 알면 이후 영상 분석에서 어느 시점에 큰 변화가 있었는지 함께 다룰 수 있습니다. 이 저장소에서는 그 준비 단계로 변화 점수를 구하고, 점수를 장면 전환 번호로 바꾸는 방식을 실험합니다. 모델 학습이나 실제 텐서 입력과의 연결은 아직 구현되어 있지 않습니다.

이 설명은 현재 코드에서 확인할 수 있는 역할을 기준으로 작성했습니다.

## 처음 읽는 순서

| 파일 | 하는 일과 필요한 이유 |
| --- | --- |
| `configs/config.yaml` | 데이터 경로와 판정 기준을 한곳에서 설정합니다. |
| `scripts/check_environment.py` | 라이브러리와 GPU, 데이터 폴더 상태를 확인합니다. |
| `scripts/create_test_video.py` | 전환 시점을 미리 아는 9초짜리 영상을 만들어 분석 결과를 비교할 수 있게 합니다. |
| `preprocessing/extract_frames.py` | 영상에서 균등 간격으로 이미지를 골라 내용을 살펴볼 수 있게 합니다. |
| `preprocessing/detect_scene_transition.py` | 이웃 프레임의 밝기 차이를 계산하고 변화가 이어지는 구간을 분류합니다. |
| `scripts/plot_scene_scores.py` | 점수 곡선과 기준선을 그려 변화 위치를 확인합니다. |
| `preprocessing/create_scene_signal.py` | 점수를 일반/급격한 전환/점진적 전환 번호로 바꾸어 CSV에 저장합니다. |
| `scripts/test_tensor_pipeline.py` | 영상과 전환 번호를 담을 배열 크기 및 CPU/GPU 배치를 가짜 데이터로 확인합니다. |

각 파이썬 파일 맨 위에는 목적과 입출력, 함수 위에는 역할, 주요 처리 부분에는 계산 이유를 주석으로 적었습니다. `#` 뒤의 설명은 파이썬이 실행하지 않습니다.

## 실행 순서

PyCharm에서 이 프로젝트의 Python 인터프리터를 선택하세요. 코드에는 OpenCV(`cv2`), NumPy, PyYAML(`yaml`), Matplotlib, PyTorch(`torch`)가 필요합니다. 아래 명령은 프로젝트 최상위 폴더의 터미널에서 실행합니다. `python`이 해당 라이브러리를 설치한 인터프리터를 가리켜야 합니다.

먼저 `configs/config.yaml`의 각 경로를 확인합니다. 현재 설정은 `D:/Graduation-project-data` 아래에 결과를 저장합니다. 같은 이름의 결과가 있으면 실행 시 덮어씁니다.

```powershell
python scripts/check_environment.py
python scripts/create_test_video.py
python preprocessing/detect_scene_transition.py
python scripts/plot_scene_scores.py
python preprocessing/create_scene_signal.py
```

그래프 창이 열리면 확인 후 닫아 다음 명령을 실행하세요. 아래 두 작업은 위 분석 흐름과 별개입니다. 이미지 추출에는 테스트 영상이 필요하고, 텐서 예제에는 영상 파일이 필요하지 않습니다.

```powershell
python preprocessing/extract_frames.py
python scripts/test_tensor_pipeline.py
```

## 입력과 결과의 연결

1. 영상 생성 → `temp/scene_transition_test.mp4`
2. 장면 탐지 → `results/scene_transition_test_scene_scores.csv` 및 콘솔의 후보 구간 목록
3. 점수 그래프 → 위 CSV를 읽어 `results/scene_transition_test_scene_scores.png` 생성
4. 장면 번호 생성 → 같은 점수 CSV를 읽어 `results/scene_transition_test_scene_signals.csv` 생성

프레임 추출은 MP4를 따로 읽어 `frames/scene_transition_test/frame_0000.jpg` 등의 파일을 만듭니다. 추출한 JPG는 현재 장면 탐지 코드의 입력이 아닙니다. 텐서 예제도 이 JPG나 CSV를 읽지 않습니다.

## 용어와 결과 읽기

- **프레임**: 영상을 구성하는 이미지 한 장입니다. 이 코드의 원본 프레임 번호는 0부터 시작합니다.
- **FPS**: 1초당 프레임 수입니다. 30 FPS 영상의 60번 프레임은 2초 위치에 해당합니다.
- **변화 점수**: 두 프레임을 회색조로 바꾼 뒤 픽셀별 밝기 차이의 절댓값을 평균 내고 255로 나눈 값입니다. 0~1 범위이며, 전환일 확률은 아닙니다.
- **임계값(threshold)**: 이 값 이상인지 비교하는 판정 기준입니다.
- **하드 컷(HARD_CUT)**: 화면이 갑자기 바뀌는 전환입니다.
- **점진적 전환(GRADUAL)**: 페이드나 디졸브처럼 여러 프레임에 걸쳐 바뀌는 전환입니다.
- **텐서**: 숫자의 다차원 배열입니다. 예제 영상 모양은 `[영상 수, 프레임 수, 채널 수, 높이, 너비] = [1, 16, 3, 224, 224]`입니다.

점수 CSV의 `frame_number=1`은 원본의 0번과 1번 프레임을 비교한 결과입니다. 첫 프레임은 비교할 이전 이미지가 없어 점수가 없습니다. 정상적으로 N장을 읽으면 점수는 N-1개 생깁니다.

장면 번호 CSV는 `scene_signal`에 **0(일반), 1(급격한 전환), 2(점진적 전환)**을 저장합니다. `scene_type`은 같은 의미의 문자열입니다. 요약의 프레임 수는 전환 사건의 개수와 다릅니다.

## 두 분류 코드의 차이

`detect_scene_transition.py`는 낮은 기준 이상인 점수를 연속 구간으로 묶습니다. 구간이 1~2프레임이고 최대 점수가 높은 기준 이상이면 `HARD_CUT`, 최소 길이 이상이면 `GRADUAL`, 나머지는 `SMALL_CHANGE`로 출력합니다.

`create_scene_signal.py`는 높은 기준 이상의 프레임을 바로 `HARD_CUT`으로 표시합니다. 낮은 기준 이상이면서 높은 기준 미만인 후보들은 잠시 모았다가 작은 점수가 나오거나 입력이 끝날 때 길이를 확인해 `GRADUAL`로 바꿉니다. 짧은 후보는 `NORMAL`로 남습니다. 현재 구현에서는 하드 컷이 나오면 바로 앞에 모인 후보를 확정하지 않고 비워, 그 후보들은 `NORMAL`로 남습니다.

따라서 콘솔의 구간 분류와 장면 번호 CSV는 항상 일치하지 않습니다. 이 주석 작업에서는 기존 판정 로직을 그대로 유지했습니다.

## 실험 범위

현재 분석은 밝기 차이를 사용하는 규칙 기반 실험입니다. 물체 움직임과 조명 변화도 전환으로 잡힐 수 있고, 밝기가 비슷한 색상 변화는 놓칠 수 있습니다. 테스트 영상의 의도된 전환 시점과 검출 결과를 비교하면서 기준값의 영향을 살펴보세요. 텐서 예제는 입력 형태 확인용이며, 16개 추출 프레임과 전체 영상의 전환 번호를 대응시키는 작업은 포함하지 않습니다.

## FaceForensics++를 D드라이브에 받기

코드는 현재 C드라이브 프로젝트에서 실행하고, 영상은 `config.yaml`의 `paths.raw` 아래 `FaceForensics++` 폴더에 저장합니다. 현재 저장 위치는 `D:/Graduation-project-data/raw/FaceForensics++`입니다.

1. [공식 이용 신청 안내](https://github.com/ondyari/FaceForensics#access)를 따라 승인받습니다.
2. 받은 공식 스크립트를 `scripts/download-FaceForensics.py`에 저장합니다. 이 프로젝트의 `download_faceforensics.py`는 공식 파일을 실행하는 별도의 도우미입니다.
3. 프로젝트 가상환경이 선택된 터미널에서 아래 명령을 실행합니다.

```powershell
# 실제 다운로드 없이 D드라이브 저장 위치와 명령 확인
python scripts/download_faceforensics.py --dry-run

# 공식 스크립트에 필요한 다운로드 진행 표시 라이브러리
python -m pip install tqdm

# 원본과 Deepfakes 영상을 각각 10개씩 c23 품질로 다운로드
python scripts/download_faceforensics.py
```

영상 개수는 `--num-videos 20`처럼 바꿀 수 있습니다. 공식 파일을 다른 위치에 두었다면 `--script "C:/경로/download-FaceForensics.py"`를 덧붙이세요. 공식 파일이 없으면 도우미는 안내만 출력하고 종료합니다.

이 도우미는 FaceForensics++를 대상으로 합니다. 구버전 FaceForensics는 승인 메일에 포함된 해당 버전의 안내를 따르세요. 기존 전처리 스크립트는 여전히 테스트 영상을 읽으며, 새 데이터셋 전체를 자동 분석하지는 않습니다.
