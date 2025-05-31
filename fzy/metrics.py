import numpy as np
from pathlib import Path
from copy import deepcopy
CURRENT_PATH = Path(__file__).parent
NONE_VALUE = -99999999


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


def load_data():
    import json
    datasets = ["activitynet", "charades", "qvhighlights", "youcook2"]
    # datasets = ["activitynet", "charades", "qvhighlights", "valor", "breakfast", "youcook2"]
    # datasets = ["activitynet", "charades", "qvhighlights", "valor", "youcook2"]
    # ds2json = lambda ds: CURRENT_PATH / "../output" /f"{ds}_output.json"
    ds2json = lambda ds: CURRENT_PATH / "../output/3b" /f"{ds}_output.json"
    ret = dict()
    for dataset in datasets:
        ret_ds = list()
        with open(ds2json(dataset), "r") as f:
            raw = json.load(f)
        if dataset == "breakfast":
            for item in raw:
                splits = item["prediction"].split(".")
                # if len(item["segments"]) != len(splits):
                #     continue
                for i in range(min(len(item["segments"]), len(splits))):
                    prediction_start = get_predictions(splits[i])
                    answer = [item["segments"][i]["start"] / 15, item["segments"][i]["end"] / 15]
                    prediction_end = prediction_start + (answer[1] - answer[0])
                    ret_ds.append({
                        "predictions": [[prediction_start, prediction_end]],
                        "ground_truths": [answer]
                    })
            continue
        for item in raw:
            prediction_start = get_predictions(item["prediction"])
            answer = item["answer"]
            # print(item)
            if isinstance(answer[0], float) or isinstance(answer[0], int):
                answer = [answer]
            else:
                assert dataset == "qvhighlights", f"data format error: {dataset}"
                answer = [answer[0]]
            prediction_end = prediction_start + (answer[0][1] - answer[0][0])
            ret_ds.append({
                "predictions": [[prediction_start, prediction_end]],
                "ground_truths": answer
            })
        ret[dataset] = ret_ds
    return ret
        

def get_predictions(pred) -> float:
    '''
    get prediction data from output json
    '''
    import re
    _pred = deepcopy(pred)
    pred = pred.strip()


    def find_first_number(s):
        match = re.search(r'\d+', s)
        if match:
            return match.group()  # 返回匹配到的第一个数字
        return None  # 如果没有数字，返回 None
    ans = find_first_number(pred)
    if ans is not None:
        return float(ans)
    
    return NONE_VALUE
    
    

def main():
    ds = load_data()
    for ds_name, ds_data in ds.items():
        print(f"{'='*10} {ds_name} {'='*10}")
        # for each dataset, calculate the average of evaluation results
        results_all = list()
        for item in ds_data:
            results = evaluate_predictions(item["predictions"], item["ground_truths"])
            # if item["predictions"][0][0] == NONE_VALUE:
                # print(f"Warning: {ds_name} has NONE_VALUE -> {results_all}")
                # continue
            results_all.append(results)
        # print(f"Total samples: {len(results_all)}")
        # calculate the average of evaluation results
        mIoU = np.mean([item["mIoU"] for item in results_all])
        recall = {k: np.mean([item["Recall"][k] for item in results_all]) for k in results_all[0]["Recall"].keys()}
        print(f"mIoU: {mIoU:.4f}")
        print(f"Recall: {recall}")
        print(f"{'='*30}")
        


def main_test():
    # 算每个query的检索结果的iou，然后计算mIoU、R@{0.3,0.5,0.7}这些指标
    # 示例测试数据
    predictions = [[0.0, 10.0], [5.0, 15.0], [20.0, 30.0]]  # 模拟预测时间段
    ground_truths = [[0.0, 10.0], [7.0, 14.0], [25.0, 35.0]]  # 模拟真值时间段

    # 计算指标
    results = evaluate_predictions(predictions, ground_truths)

    # 打印结果
    print("Evaluation Results:")
    print(f"mIoU: {results['mIoU']:.4f}")
    for threshold, recall in results["Recall"].items():
        print(f"R@{threshold}: {recall:.4f}")

# 测试数据
if __name__ == "__main__":
    main()
    # main_test()


# ========== activitynet ==========
# mIoU: 0.1764
# Recall: {0.3: 0.24285400175901495, 0.5: 0.1774406332453826, 0.7: 0.11015831134564644}
# ==============================
# ========== charades ==========
# mIoU: 0.1186
# Recall: {0.3: 0.16102150537634408, 0.5: 0.1174731182795699, 0.7: 0.07956989247311828}
# ==============================
# ========== qvhighlights ==========
# mIoU: 0.1982
# Recall: {0.3: 0.26362625139043383, 0.5: 0.22024471635150167, 0.7: 0.14349276974416017}
# ==============================


# 忽略掉所有无法解析时间的数据以提高准确性
# ========== activitynet ==========
# mIoU: 0.1793
# Recall: {0.3: 0.24684322270644765, 0.5: 0.18035534696614147, 0.7: 0.11196781763325511}
# ==============================
# ========== charades ==========
# mIoU: 0.2803
# Recall: {0.3: 0.380559085133418, 0.5: 0.2776365946632783, 0.7: 0.1880559085133418}
# ==============================
# ========== qvhighlights ==========
# mIoU: 0.1996
# Recall: {0.3: 0.26539753639417696, 0.5: 0.22172452407614782, 0.7: 0.1444568868980963}