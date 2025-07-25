import json
import jsonlines
# import dataset from ling99/OCRBench_v2
import os
from datasets import load_dataset
def results_ok(gt, pred):
    # Check if the predicted text matches the ground truth text
    if str(gt).strip() in str(pred).strip():
        return True
    if str(pred).strip() in str(gt).strip():
        return True
    # Check if the predicted text is a substring of the ground truth text
    return False


def eval_ocrbenchv2():
# Load the OCRBench_v2 dataset
    results_path = "/home1/hxl/disk/EAGLE/output/eval/20250608/ocrbv2.json"
    with open(results_path, "r") as f:
        results = json.load(f)
    dataset = load_dataset("ling99/OCRBench_v2", split="test")
    
    all_results = []
    for r in results:
        doc_id = r["doc_id"]
        # find the corresponding document in the dataset
        doc = dataset[int(doc_id)]
        assert doc["id"] == doc_id, f"Document ID mismatch: {doc['id']} != {doc_id}"
        if results_ok(doc['answers'], r['text_output']):
            all_results.append(1)
        else:
            all_results.append(0)
    # Calculate the accuracy
    accuracy = sum(all_results) / len(all_results)
    print(f"Accuracy: {accuracy:.4f}")
    

def eval_chartqa():
    # results_path = "/home1/hxl/disk/EAGLE/output/eval/pr_llm__finetune-image-llama3.2-3b-fzy-doc-chart-before/20250610_233221_samples_chartqa.jsonl"
    results_path = "/home1/hxl/disk2/Backup/EAGLE/output/eval/finetune-image-llama3.2-3b-fzy-qwen2vl-batch-llava-eagle__checkpoint-20000/20250706_223701_samples_chartqa.jsonl"
    with jsonlines.open(results_path, "r") as f:
        results = [r for r in f]
    # dataset = load_dataset("lmms-lab/ChartQA", split="test")
    # print(results[2000]['doc']['question'], dataset[2000]['question'])
    # for doc_id, item in enumerate(dataset):
    #     # find the item according to doc_id in results
    #     target_result = [r for r in results if r['doc_id'] == doc_id][0]
        
    # 好像只需要通过results就够了
    all_results = []
    for item in results:
        gt = item['doc']['answer']
        preds = item['filtered_resps']
        # 只要任意一个答案正确即可
        correct = any(results_ok(gt, pred) for pred in preds)
        all_results.append(correct)
    # Calculate the accuracy
    accuracy = sum(all_results) / len(all_results)
    print(f"ChartQA Accuracy: {accuracy:.4f}")
        
        

def eval_docvqa():
    results_path = "/home1/hxl/disk/EAGLE/output/eval/pr_llm__finetune-image-llama3.2-3b-fzy-doc-chart-before/20250610_233221_samples_docvqa_val.jsonl"
    with jsonlines.open(results_path, "r") as f:
        results = [r for r in f]
    # Load the DocVQA dataset
    # dataset = load_dataset("lmms-lab/DocVQA", 'DocVQA', split="validation")
    
    all_results = []
    for item in results:
        gts = item['doc']['answers']
        pred = item['filtered_resps'][0]
        # 只要任意一个答案正确即可
        correct = any(results_ok(gt, pred) for gt in gts)
        all_results.append(correct)
    # Calculate the accuracy
    accuracy = sum(all_results) / len(all_results)
    print(f"DocVQA Accuracy: {accuracy:.4f}")
    
    
    
def eval_textvqa():
    results_path = "/home1/hxl/disk/EAGLE/output/eval/pr_llm__finetune-image-llama3.2-3b-fzy-doc-chart-before/20250610_233221_samples_textvqa_val.jsonl"
    with jsonlines.open(results_path, "r") as f:
        results = [r for r in f]
    # Load the TextVQA dataset
    # dataset = load_dataset("lmms-lab/textvqa", split="validation")
    
    all_results = []
    for item in results:
        gts = item['doc']['answers']
        pred = item['filtered_resps'][0]
        # 只要任意一个答案正确即可
        correct = any(results_ok(gt, pred) for gt in gts)
        all_results.append(correct)
    # Calculate the accuracy
    accuracy = sum(all_results) / len(all_results)
    print(f"TextVQA Accuracy: {accuracy:.4f}")


def eval_mme():
    # results_path = "/home1/hxl/disk2/Backup/EAGLE/output/eval/finetune-image-llama3.2-3b-fzy-qwen2vl-batch-llava-eagle__checkpoint-20000/20250706_231251_samples_mme.jsonl"
    # results_path = "/home1/hxl/disk2/Backup/EAGLE/output/eval/en_pr__finetune-eagle-x1-llama3.2-3b-image_L/20250518_122849_samples_mme.jsonl"
    # results_path = "/home1/hxl/disk2/Backup/EAGLE/output/eval/pr_llm__finetune-image-llama3.2-3b-fzy-doc-chart/20250607_190641_samples_mme.jsonl"
    # results_path = "/home1/hxl/disk2/Backup/EAGLE/output/eval/image__finetune-eagle-x1-llama3.2-1b-image_L/20241204_164823_samples_mme.jsonl"
    results_path = "/home1/hxl/disk2/Backup/EAGLE/output/eval/pr_llm__finetune-eagle-x1-llama3.2-3b-image_L/20250707_132631_samples_mme.jsonl"
    with jsonlines.open(results_path, "r") as f:
        results = [r for r in f]
    # Load the MME dataset
    # dataset = load_dataset("lmms-lab/mme", split="validation")
    all_results = {
        'mme_perception_score': 0,
        'mme_cognition_score': 0,
    }
    # item:
    # {
    #     "doc_id": 0,
    #     "doc": {
    #         "question_id": "code_reasoning/0020.png",
    #         "question": "Is a python code shown in the picture? Please answer yes or no.",
    #         "answer": "Yes",
    #         "category": "code_reasoning"
    #     },
    #     "target": "Yes",
    #     "arguments": {
    #         "0": "max_new_tokens",
    #         "1": "temperature",
    #         "2": "top_p",
    #         "3": "num_beams",
    #         "4": "do_sample",
    #         "5": "image_sizes"
    #     },
    #     "resps": [
    #         [
    #             "Yes"
    #         ]
    #     ],
    #     "filtered_resps": [
    #         "Yes"
    #     ],
    #     "doc_hash": "74234e98afe7498fb5daf1f36ac2d78acc339464f950703b8c019892f982b90b",
    #     "prompt_hash": "06b9f4c644cdd6f3fb1dc59e2bc74fcf78c6a96d0527ef619968f1f5deba1b64",
    #     "target_hash": "85a39ab345d672ff8ca9b9c6876f3adcacf45ee7c1e2dbd2408fd338bd55e07e",
    #     "mme_cognition_score": {
    #         "question_id": "code_reasoning/0020.png",
    #         "category": "code_reasoning",
    #         "score": 1.0
    #     },
    #     "input": "Is a python code shown in the picture?\nAnswer the question using a single word or phrase."
    # }
    results_dict = dict()
    for item in results:
        question_id = item["doc"]['question_id']
        if question_id not in results_dict:
            results_dict[question_id] = list()
        results_dict[question_id].append(item)
    # filter the items in results_dict if the length of the list is not 2, print the count
    filtered_results_dict = {k: v for k, v in results_dict.items() if len(v) == 2}
    print(f"Filtered results count from {len(results_dict)} to {len(filtered_results_dict)}")

        




    
    # for item in results:
    #     gt = item['target']
    #     filtered_resps = item['filtered_resps'][0]
    #     if gt.lower() in filtered_resps.lower():
    #         if 'mme_perception_score' in item:
    #             # all_results['mme_perception_score'] += 1
    #             all_results['mme_perception_score'] += item['mme_perception_score']['score']
    #         if 'mme_cognition_score' in item:
    #             # all_results['mme_cognition_score'] += 1
    #             all_results['mme_cognition_score'] += item['mme_cognition_score']['score']
    # Calculate the accuracy
    print(all_results)
        


if __name__ == "__main__":
    eval_ocrbenchv2()
    # eval_chartqa()
    # eval_docvqa()
    # eval_textvqa()
    # eval_mme()

