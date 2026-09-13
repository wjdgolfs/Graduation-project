# ============================================================================
# 얼굴 영역 계산 도구
# ----------------------------------------------------------------------------
# MediaPipe 가 찾은 얼굴 랜드마크(478점)로 다음을 계산하는 함수들입니다.
#   - 크롭 상자   : 얼굴을 넉넉히 둘러싼 정사각형 상자, 그리고 프레임 사이 떨림을 줄이는 이동 평균
#   - 얼굴 크롭   : 상자 부분을 잘라 정해진 크기(예: 240x240)로 맞춘 이미지와, 그 좌표계로 옮긴 랜드마크
#   - 부위 마스크 : 눈·입·얼굴 윤곽의 볼록 껍질, 얼굴 경계 띠
#   - 배경 마스크 : 얼굴 주변을 넓게 뺀 나머지 영역, 그리고 배경의 프레임 간 변화량
#
# 좌표 규칙
#   점과 상자는 (x, y) = (가로, 세로) 픽셀 좌표입니다. 상자는 [x1, y1, x2, y2] (왼쪽 위, 오른쪽 아래).
#   numpy 이미지 배열은 [세로, 가로] 순서라 image[y, x] 로 접근합니다. 둘을 헷갈리기 쉬우니 주의하세요.
#
# MediaPipe 에 의존하는 함수는 region_point_indices 하나뿐이라, 나머지는 mediapipe 없이도 테스트할 수 있습니다.
# 테스트: tests/test_face_regions.py
# ============================================================================

import cv2
import numpy as np


# MediaPipe 가 제공하는 연결 정보(선분 목록)에서 부위마다 쓰이는 점 번호를 모읍니다.
#   반환 예: {"face_oval": [10, 21, 54, ...], "left_eye": [...], "right_eye": [...], "lips": [...]}
#   left_eye 는 "사진 속 사람의 왼쪽 눈"이라 이미지에서는 오른쪽에 보입니다.
# mediapipe 는 불러오는 데 시간이 걸려서 이 함수를 부를 때만 import 합니다.
def region_point_indices():
    from mediapipe.tasks.python import vision

    connections = vision.FaceLandmarksConnections
    groups = {
        "face_oval": connections.FACE_LANDMARKS_FACE_OVAL,
        "left_eye": connections.FACE_LANDMARKS_LEFT_EYE,
        "right_eye": connections.FACE_LANDMARKS_RIGHT_EYE,
        "lips": connections.FACE_LANDMARKS_LIPS,
    }
    # 선분(start, end) 양 끝의 점 번호를 집합으로 모아 중복을 없앱니다.
    return {
        name: sorted({index for connection in items for index in (connection.start, connection.end)})
        for name, items in groups.items()
    }


# 점들을 모두 감싸는 가장 작은 상자 [x1, y1, x2, y2] 를 돌려줍니다.
# nan 인 점은 무시하고, 쓸 수 있는 점이 없으면 nan 상자를 돌려줍니다.
def tight_box(points):
    points = np.asarray(points, dtype=np.float64)
    valid = points[np.isfinite(points).all(axis=1)]
    if len(valid) == 0:
        return np.full(4, np.nan)
    x1, y1 = valid.min(axis=0)
    x2, y2 = valid.max(axis=0)
    return np.array([x1, y1, x2, y2])


# 두 상자가 겹친 넓이 ÷ 두 상자를 합친 넓이(IoU). 0 이면 안 겹침, 1 이면 완전히 같습니다.
def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


# 상자를 같은 중심의 정사각형으로 바꾸고, 가장 긴 변의 scale 배로 키웁니다.
#   예) 가로 100, 세로 120 인 상자, scale 1.3 → 한 변 156 인 정사각형
# box 는 [4] 하나여도 되고 [프레임 수, 4] 여러 개여도 됩니다("..." 는 앞쪽 축을 모두 그대로 둔다는 뜻).
def square_box(box, scale):
    box = np.asarray(box, dtype=np.float64)
    center_x = (box[..., 0] + box[..., 2]) / 2
    center_y = (box[..., 1] + box[..., 3]) / 2
    side = np.maximum(box[..., 2] - box[..., 0], box[..., 3] - box[..., 1]) * scale
    half = side / 2
    return np.stack([center_x - half, center_y - half, center_x + half, center_y + half], axis=-1)


# 프레임마다의 상자 [F, 4] 를 시간 방향으로 부드럽게 만듭니다.
#   1) 얼굴을 못 찾은 프레임(nan)은 가장 가까운 "찾은 프레임"의 상자로 채웁니다.
#   2) 상자의 중심과 크기에 창 크기 window 인 이동 평균을 적용합니다(가운데 정렬, 양 끝은 있는 만큼만 평균).
# 왜 필요한가
#   프레임마다 상자가 몇 픽셀씩 흔들리면 크롭 이미지 전체가 흔들려서, 조작이 없어도 "프레임 사이 변화"가 생깁니다.
#   시간적 불연속을 보는 이 연구에서는 그 떨림이 잡음이 됩니다.
# 반환값: (부드럽게 만든 상자 [F, 4], 원래 얼굴을 찾았는지 [F] bool). 한 프레임도 못 찾았으면 상자는 모두 nan 입니다.
def smooth_boxes(boxes, window):
    boxes = np.asarray(boxes, dtype=np.float64)
    found = np.isfinite(boxes).all(axis=1)
    if not found.any():
        return boxes.copy(), found

    frames = len(boxes)
    valid_index = np.flatnonzero(found)
    # [프레임 수, 찾은 프레임 수] 거리 표를 만들어, 프레임마다 가장 가까운 "찾은 프레임" 번호를 고릅니다.
    distance = np.abs(np.arange(frames)[:, None] - valid_index[None, :])
    filled = boxes[valid_index[distance.argmin(axis=1)]]

    if window <= 1:
        return filled, found

    center = np.stack([(filled[:, 0] + filled[:, 2]) / 2, (filled[:, 1] + filled[:, 3]) / 2], axis=1)
    size = np.stack([filled[:, 2] - filled[:, 0], filled[:, 3] - filled[:, 1]], axis=1)
    half_window = window // 2
    smoothed_center = np.empty_like(center)
    smoothed_size = np.empty_like(size)
    for t in range(frames):
        lo, hi = max(0, t - half_window), min(frames, t + half_window + 1)
        smoothed_center[t] = center[lo:hi].mean(axis=0)
        smoothed_size[t] = size[lo:hi].mean(axis=0)

    half = smoothed_size / 2
    # [cx - w/2, cy - h/2] 와 [cx + w/2, cy + h/2] 를 이어 붙이면 [x1, y1, x2, y2] 가 됩니다.
    return np.concatenate([smoothed_center - half, smoothed_center + half], axis=1), found


# 실수 상자를 픽셀 단위 정사각형 (왼쪽 위 x, 왼쪽 위 y, 한 변) 으로 바꿉니다.
# crop_frame 과 to_crop_coords 가 똑같이 반올림해야 크롭 이미지와 랜드마크 좌표가 정확히 맞습니다.
def integer_square(box):
    x1 = int(round(float(box[0])))
    y1 = int(round(float(box[1])))
    side = max(1, int(round(float(box[2]) - float(box[0]))))
    return x1, y1, side


# 점들을 크롭 좌표로 옮깁니다. nan 은 nan 으로 남습니다.
#   크롭 좌표 = (원래 좌표 - 상자 왼쪽 위) x (크롭 크기 / 상자 한 변)
def to_crop_coords(points, box, size):
    x1, y1, side = integer_square(box)
    return (np.asarray(points, dtype=np.float64) - np.array([x1, y1])) * (size / side)


# 프레임에서 정사각형 상자 부분을 잘라 size x size 로 맞춥니다.
# 상자가 프레임 밖으로 나가면 나간 부분은 검정(0)으로 채웁니다(얼굴이 화면 가장자리에 있을 때).
# 줄일 때는 INTER_AREA(계단 현상이 적음), 키울 때는 INTER_CUBIC 을 씁니다.
def crop_frame(frame, box, size):
    height, width = frame.shape[:2]
    x1, y1, side = integer_square(box)
    x2, y2 = x1 + side, y1 + side

    # 먼저 한 변 side 인 검은 캔버스를 만들고, 프레임 안에 들어오는 부분만 복사합니다.
    canvas = np.zeros((side, side, frame.shape[2]), dtype=frame.dtype)
    src_x1, src_y1 = max(0, x1), max(0, y1)
    src_x2, src_y2 = min(width, x2), min(height, y2)
    if src_x1 < src_x2 and src_y1 < src_y2:
        canvas[src_y1 - y1:src_y2 - y1, src_x1 - x1:src_x2 - x1] = frame[src_y1:src_y2, src_x1:src_x2]

    interpolation = cv2.INTER_AREA if side > size else cv2.INTER_CUBIC
    return cv2.resize(canvas, (size, size), interpolation=interpolation)


# 점 번호 indices 에 해당하는 점들의 볼록 껍질(점들을 고무줄로 감싼 모양)을 채운 마스크를 만듭니다.
# 반환값: [height, width] uint8 (안쪽 1, 바깥 0). 쓸 수 있는 점이 3개 미만이면 모두 0 입니다.
def hull_mask(points, indices, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    selected = np.asarray(points, dtype=np.float64)[list(indices)]
    selected = selected[np.isfinite(selected).all(axis=1)]
    if len(selected) < 3:
        return mask
    hull = cv2.convexHull(np.round(selected).astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 1)
    return mask


# 얼굴 윤곽을 따라가는 띠 모양 마스크입니다.
# 윤곽 볼록 껍질을 band 픽셀만큼 키운 것(dilate)에서 줄인 것(erode)을 빼면, 윤곽선 양쪽으로 band 픽셀씩 두꺼운 띠가 남습니다.
# face-swap 이 가짜 얼굴과 원래 영상을 섞는 경계가 이 띠 안에 들어옵니다.
def boundary_band_mask(points, oval_indices, height, width, band):
    face = hull_mask(points, oval_indices, height, width)
    if band <= 0 or not face.any():
        return np.zeros_like(face)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
    return cv2.dilate(face, kernel) - cv2.erode(face, kernel)


# 배경 마스크: 얼굴 상자를 exclusion_scale 배로 넓힌 정사각형의 바깥을 True 로 둡니다.
#   box 는 이 마스크와 같은 좌표계(예: 축소판)의 얼굴 상자입니다. nan 이면 전체를 배경으로 봅니다.
#   머리카락·목·어깨는 사람과 함께 움직여 "장면 전체의 변화"를 대표하지 못하므로 넉넉히 뺍니다.
def background_mask(height, width, box, exclusion_scale):
    mask = np.ones((height, width), dtype=bool)
    box = np.asarray(box, dtype=np.float64)
    if not np.isfinite(box).all():
        return mask
    x1, y1, x2, y2 = square_box(box, exclusion_scale)
    # floor(내림)·ceil(올림)으로 경계 픽셀까지 확실히 뺍니다.
    x1, y1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
    x2, y2 = min(width, int(np.ceil(x2))), min(height, int(np.ceil(y2)))
    if x1 < x2 and y1 < y2:
        mask[y1:y2, x1:x2] = False
    return mask


# 상자를 프레임 안으로 자른 정수 창 (x1, y1, x2, y2) 로 바꿉니다. 프레임 밖으로 나간 부분은 잘려 나갑니다.
def clip_window(box, height, width):
    x1 = max(0, int(np.floor(box[0])))
    y1 = max(0, int(np.floor(box[1])))
    x2 = min(width, int(np.ceil(box[2])))
    y2 = min(height, int(np.ceil(box[3])))
    return x1, y1, x2, y2


# 프레임 전체를 겹치게 나눈 정사각형 창 목록 [(x1, y1, x2, y2), ...] 을 만들고, 화면 가운데에 가까운 창부터 돌려줍니다.
#   창의 한 변 = 프레임 짧은 변 x window_fraction, 이웃한 창은 한 변의 (1 - overlap) 만큼씩 떨어집니다.
#   마지막 창이 오른쪽/아래 끝에 딱 붙도록 하나를 더 넣어, 프레임의 모든 픽셀이 적어도 한 창에 들어갑니다.
# 왜 필요한가
#   MediaPipe 얼굴 검출기는 얼굴이 화면을 크게 차지할 때 잘 찾습니다. Celeb-DF 처럼 얼굴 폭이 화면의 10% 정도면
#   전체 프레임에서는 못 찾고(실패 영상에서 30프레임 중 0장), 얼굴 주변을 잘라 넣으면 찾습니다(30장 모두).
#   인터뷰 영상은 얼굴이 대개 가운데 있어서 가운데 창부터 시도하면 빨리 찾습니다.
def search_windows(height, width, window_fraction=0.5, overlap=0.5):
    side = max(1, int(round(min(height, width) * window_fraction)))
    stride = max(1, int(round(side * (1 - overlap))))

    xs = list(range(0, max(1, width - side + 1), stride))
    ys = list(range(0, max(1, height - side + 1), stride))
    if xs[-1] != max(0, width - side):
        xs.append(max(0, width - side))
    if ys[-1] != max(0, height - side):
        ys.append(max(0, height - side))

    windows = [(x, y, x + side, y + side) for y in ys for x in xs]
    center_x, center_y = width / 2, height / 2
    # 창 중심과 화면 중심 사이 거리의 제곱으로 정렬합니다(제곱근은 순서에 영향이 없어 생략).
    windows.sort(key=lambda w: ((w[0] + w[2]) / 2 - center_x) ** 2 + ((w[1] + w[3]) / 2 - center_y) ** 2)
    return windows


# 두 흑백 이미지의 픽셀 차이 절댓값을 mask 가 True 인 곳에서만 평균내고 255 로 나눕니다(0~1).
# mask 에 True 가 하나도 없으면 nan 입니다.
# uint8 끼리 빼면 음수가 255 근처로 돌아가는(넘침) 문제가 있어 int16 으로 바꿔서 뺍니다.
def masked_mean_abs_diff(previous, current, mask):
    if not mask.any():
        return float("nan")
    difference = np.abs(current.astype(np.int16) - previous.astype(np.int16))
    return float(difference[mask].mean() / 255.0)
