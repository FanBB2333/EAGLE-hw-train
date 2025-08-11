import numpy as np
import re
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
    if len(ground_truths) > len(predictions):
        ground_truths = ground_truths[:len(predictions)]
    assert len(predictions) == len(ground_truths), f"预测结果: {predictions}, 和真值: {ground_truths} 数量不一致"

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
    # datasets = ["activitynet", "charades", "qvhighlights", "youcook2"]
    # datasets = ["activitynet", "charades", "qvhighlights", "youcook2"]
    datasets = ["mvbench"]
    # datasets = ["activitynet", "charades", "qvhighlights", "valor", "breakfast", "youcook2"]
    # datasets = ["activitynet", "charades", "qvhighlights", "valor", "youcook2"]
    # ds2json = lambda ds: CURRENT_PATH / "../output" /f"{ds}_output.json"
    ds2json = lambda ds: CURRENT_PATH / "../output/3b" /f"{ds}_output.json"
    ret = dict()
    for dataset in datasets:
        ret_ds = list()
        with open(ds2json(dataset), "r") as f:
            raw = json.load(f)
        if dataset == "mvbench":
            ret_ds = raw
            ret[dataset] = ret_ds
            continue
        elif dataset == "breakfast":
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


def get_last_number(pred) -> float:
    '''
    get the last number from prediction text for youcook2 dataset
    '''
    import re
    pred = pred.strip()
    
    # 找到所有数字
    matches = re.findall(r'\d+', pred)
    if matches:
        return float(matches[-1])  # 返回最后一个数字
    
    return NONE_VALUE
    

def eval_mvbench(res):
    def eq(pred, gt):
        # pred, gt = pred.lower().strip(), gt.lower().strip()
        # if pred == gt or pred in gt or gt in pred:
        #     return True
        # find pred with (x)
        if "(" in pred and ")" in pred:
            pred_choice = pred.split("(")[1].strip()
            if len(pred_choice) != 0:
                pred_choice = pred_choice[0]
            gt_choice = gt.split("(")[1][0]
            # print(f"gt: {gt}, gt_choice: {gt_choice}")
            if pred_choice == gt_choice or pred_choice in gt_choice or gt_choice in pred_choice:
                # print(f"pred_choice: {pred_choice}, gt_choice: {gt_choice}")
                return True
        pred_splits = pred.split()
        # filter the stop words
        pred_splits = [ps for ps in pred_splits if ps not in ["the", "a", "an", "is", "are", "was", "were", "to", "of"]]
        for ps in pred_splits:
            if ps in gt or gt in ps:
                return True
        return False
    # {
    #     "task": "mvbench",
    #     "data_path": "/home6/fzy/EAGLE/dataset/MVBench/star/Charades_segment/USNP1_1.2999999999999998_15.4.mp4",
    #     "question": "Question: Which object was taken by the person?\nOptions:\n(A) The clothes.\n(B) The pillow.\n(C) The shoe.\n(D) The phone/camera.\nOnly give the best option and do not explain why.\n",
    #     "answer": "(D) The phone/camera.",
    #     "class": "object_interaction",
    #     "prediction": "The best option is (D) The phone/camera."
    # },
    all_res = list()
    for item in res:
        # print(f"prediction: {item['prediction']}, answer: {item['answer']}")
        all_res.append(eq(item["prediction"], item["answer"]))
    all_res = np.array(all_res)
    acc = np.mean(all_res)
    print(f"mvbench acc: {acc:.4f}")
    return None


def evaluate_inference_results(inference_results: list, dataset_name: str) -> dict:
    """
    Evaluate inference results for different datasets
    
    Args:
        inference_results: List of dictionaries containing prediction results
                          Each dict should have 'prediction', 'answer', etc.
        dataset_name: Name of the dataset ('activitynet', 'charades', 'qvhighlights', 
                     'youcook2', 'breakfast', 'valor', 'mvbench')
    
    Returns:
        Dictionary containing evaluation metrics
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == "mvbench":
        return eval_mvbench_from_results(inference_results)
    elif dataset_name in ["activitynet", "charades", "qvhighlights", "youcook2", "breakfast", "valor"]:
        return eval_temporal_localization_from_results(inference_results, dataset_name)
    else:
        return {
            "error": f"Unsupported dataset: {dataset_name}",
            "supported_datasets": ["activitynet", "charades", "qvhighlights", "youcook2", "breakfast", "valor", "mvbench"]
        }


def eval_mvbench_from_results(inference_results: list) -> dict:
    """
    Evaluate MVBench results from inference output
    
    Args:
        inference_results: List of inference results
    
    Returns:
        Dictionary containing accuracy metric
    """
    def eq(pred, gt):
        # find pred with (x)
        if "(" in pred and ")" in pred:
            pred_choice = pred.split("(")[1].strip()
            if len(pred_choice) != 0:
                pred_choice = pred_choice[0]
            gt_choice = gt.split("(")[1][0]
            if pred_choice == gt_choice or pred_choice in gt_choice or gt_choice in pred_choice:
                return True
        pred_splits = pred.split()
        # filter the stop words
        pred_splits = [ps for ps in pred_splits if ps not in ["the", "a", "an", "is", "are", "was", "were", "to", "of"]]
        for ps in pred_splits:
            if ps in gt or gt in ps:
                return True
        return False
    
    all_res = list()
    total_samples = len(inference_results)
    
    for item in inference_results:
        prediction = item.get("prediction", "")
        answer = item.get("answer", "")
        all_res.append(eq(prediction, answer))
    
    all_res = np.array(all_res)
    acc = np.mean(all_res)
    
    return {
        "dataset": "mvbench",
        "total_samples": total_samples,
        "accuracy": float(acc),
        "correct_predictions": int(np.sum(all_res)),
        "details": {
            "metric_type": "accuracy",
            "description": "Multiple choice accuracy for video understanding"
        }
    }


def eval_temporal_localization_from_results(inference_results: list, dataset_name: str) -> dict:
    """
    Evaluate temporal localization results from inference output
    
    Args:
        inference_results: List of inference results
        dataset_name: Name of the dataset
    
    Returns:
        Dictionary containing mIoU and Recall metrics
    """
    processed_data = []
    invalid_predictions = 0
    
    for item in inference_results:
        prediction_text = item.get("prediction", "")
        answer = item.get("answer", [])
        
        # Extract prediction start time - use different extraction methods for different datasets
        if dataset_name == "youcook2":
            prediction_start = get_last_number(prediction_text)  # Use last number for youcook2
        else:
            prediction_start = get_predictions(prediction_text)  # Use first number for other datasets
        
        if prediction_start == NONE_VALUE:
            invalid_predictions += 1
            continue
        
        # Handle different answer formats
        if isinstance(answer, (int, float)):
            # Single number case
            ground_truth = [answer]
        elif isinstance(answer, list) and len(answer) > 0:
            if isinstance(answer[0], (int, float)):
                # [start, end] format
                ground_truth = [answer]
            else:
                # [[start, end], ...] format
                ground_truth = answer
        else:
            invalid_predictions += 1
            continue
        
        # For breakfast dataset, handle segments differently
        if dataset_name == "breakfast" and "segments" in item:
            segments = item["segments"]
            prediction_splits = prediction_text.split(".")
            
            for i in range(min(len(segments), len(prediction_splits))):
                seg_prediction_start = get_predictions(prediction_splits[i])
                if seg_prediction_start == NONE_VALUE:
                    continue
                seg_answer = [segments[i]["start"] / 15, segments[i]["end"] / 15]
                seg_prediction_end = seg_prediction_start + (seg_answer[1] - seg_answer[0])
                
                processed_data.append({
                    "predictions": [[seg_prediction_start, seg_prediction_end]],
                    "ground_truths": [seg_answer]
                })
            continue
        
        # For other datasets
        if len(ground_truth) > 0 and len(ground_truth[0]) == 2:
            # Calculate prediction end time based on ground truth duration
            prediction_end = prediction_start + (ground_truth[0][1] - ground_truth[0][0])
            
            processed_data.append({
                "predictions": [[prediction_start, prediction_end]],
                "ground_truths": ground_truth
            })
    
    if not processed_data:
        return {
            "dataset": dataset_name,
            "error": "No valid predictions found",
            "total_samples": len(inference_results),
            "invalid_predictions": invalid_predictions
        }
    
    # Calculate metrics for all valid samples
    results_all = []
    for item in processed_data:
        results = evaluate_predictions(item["predictions"], item["ground_truths"])
        results_all.append(results)
    
    # Calculate average metrics
    mIoU = np.mean([item["mIoU"] for item in results_all])
    recall_thresholds = results_all[0]["Recall"].keys() if results_all else [0.3, 0.5, 0.7]
    recall = {k: np.mean([item["Recall"][k] for item in results_all]) for k in recall_thresholds}
    
    return {
        "dataset": dataset_name,
        "total_samples": len(inference_results),
        "valid_samples": len(processed_data),
        "invalid_predictions": invalid_predictions,
        "mIoU": float(mIoU),
        "Recall": {k: float(v) for k, v in recall.items()},
        "details": {
            "metric_type": "temporal_localization",
            "description": "IoU-based temporal localization metrics",
            "thresholds": list(recall_thresholds)
        }
    }


def main():
    ds = load_data()
    for ds_name, ds_data in ds.items():
        print(f"{'='*10} {ds_name} {'='*10}")
        if ds_name == "mvbench":
            eval_mvbench(ds_data)
            continue
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