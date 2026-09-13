# ============================================================================
# 이진 분류 성능 지표
# ----------------------------------------------------------------------------
# fake(라벨 1)를 양성(positive)으로 둡니다. probabilities 는 영상마다 "fake 일 확률"입니다.
#   TP: fake 를 fake 로 맞힘      FP: real 을 fake 로 잘못 봄
#   TN: real 을 real 로 맞힘      FN: fake 를 real 로 놓침
#   정확도(accuracy)  = (TP + TN) / 전체
#   정밀도(precision) = TP / (TP + FP)    fake 라고 판정한 것 중 진짜 fake 비율
#   재현율(recall)    = TP / (TP + FN)    진짜 fake 중 찾아낸 비율
#   F1               = 2 x 정밀도 x 재현율 / (정밀도 + 재현율)
#   AUC              = fake 하나와 real 하나를 무작위로 골랐을 때 fake 점수가 더 높을 확률(점수가 같으면 절반으로 셈).
#                      기준값(threshold)과 상관없이 점수의 순서만으로 정해집니다.
# 정확도·정밀도·재현율·F1 은 원 논문 식 (14)~(17) 과 같은 정의입니다.
# ============================================================================

import numpy as np
from scipy.stats import mannwhitneyu


# 반환값: accuracy, precision, recall, f1, auc, tp, fp, tn, fn 을 담은 딕셔너리
def classification_metrics(labels, probabilities, threshold=0.5):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    # 출력 노드 2개 중 큰 쪽을 고르는 판정은 "fake 확률 >= 0.5" 와 같습니다.
    predictions = (probabilities >= threshold).astype(int)

    tp = int(((predictions == 1) & (labels == 1)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    # Mann-Whitney U 통계량은 "fake 점수 > real 점수"인 쌍의 수(같으면 0.5)라서, 쌍의 총수로 나누면 AUC 입니다.
    fake_scores = probabilities[labels == 1]
    real_scores = probabilities[labels == 0]
    auc = float("nan")
    if len(fake_scores) and len(real_scores):
        auc = mannwhitneyu(fake_scores, real_scores).statistic / (len(fake_scores) * len(real_scores))

    return {
        "accuracy": (tp + tn) / len(labels),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "auc": float(auc),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


# 지표를 한 줄로 보기 좋게 만듭니다.
def format_metrics(metrics):
    return (
        f"정확도 {metrics['accuracy']:.3f} | 정밀도 {metrics['precision']:.3f} | 재현율 {metrics['recall']:.3f} | "
        f"F1 {metrics['f1']:.3f} | AUC {metrics['auc']:.3f} | "
        f"TP {metrics['tp']} FP {metrics['fp']} TN {metrics['tn']} FN {metrics['fn']}"
    )
