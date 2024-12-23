import numpy as np

def calculate_iou(pred, gt):
    """
    计算单个预测与真值的 IoU。
    :param pred: tuple, (start_time, end_time) 预测时间段
    :param gt: tuple, (start_time, end_time) 实际时间段
    :return: float, IoU 值
    """
    pred_start, pred_end = pred
    gt_start, gt_end = gt

    intersection_start = max(pred_start, gt_start)
    intersection_end = min(pred_end, gt_end)
    intersection = max(0, intersection_end - intersection_start)

    union_start = min(pred_start, gt_start)
    union_end = max(pred_end, gt_end)
    union = union_end - union_start

    if union == 0:
        return 0
    return intersection / union


def evaluate_predictions(predictions, ground_truths, thresholds=[0.3, 0.5, 0.7]):
    """
    评估检索结果。
    :param predictions: list of tuples, 每个元素为 (start_time, end_time) 的预测时间段
    :param ground_truths: list of tuples, 每个元素为 (start_time, end_time) 的真值时间段
    :param thresholds: list of floats, IoU 阈值列表
    :return: dict, 包含 mIoU 和 R@ 的指标
    """
    assert len(predictions) == len(ground_truths), "预测结果和真值数量不一致"

    iou_scores = []
    recall_scores = {threshold: 0 for threshold in thresholds}

    # 逐个计算 IoU
    for pred, gt in zip(predictions, ground_truths):
        iou = calculate_iou(pred, gt)
        iou_scores.append(iou)

        # 计算各个阈值下的召回
        for threshold in thresholds:
            if iou >= threshold:
                recall_scores[threshold] += 1

    # 计算 mIoU
    mIoU = np.mean(iou_scores)

    # 计算 Recall
    total_samples = len(predictions)
    for threshold in thresholds:
        recall_scores[threshold] /= total_samples

    # 返回指标
    return {
        "mIoU": mIoU,
        "Recall": recall_scores
    }


# 测试数据
if __name__ == "__main__":
    # 示例测试数据
    predictions = [(0.0, 10.0), (5.0, 15.0), (20.0, 30.0)]  # 模拟预测时间段
    ground_truths = [(0.0, 10.0), (7.0, 14.0), (25.0, 35.0)]  # 模拟真值时间段

    # 计算指标
    results = evaluate_predictions(predictions, ground_truths)

    # 打印结果
    print("Evaluation Results:")
    print(f"mIoU: {results['mIoU']:.4f}")
    for threshold, recall in results["Recall"].items():
        print(f"R@{threshold}: {recall:.4f}")
