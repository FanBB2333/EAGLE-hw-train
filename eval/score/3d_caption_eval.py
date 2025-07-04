import json
import argparse
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

parser = argparse.ArgumentParser()
parser.add_argument(
    "--result_file",
    type=str,
    required=True,
    help="Path to result file with ground truth.",
)
args = parser.parse_args()

with open(args.result_file, 'r') as f:
    results_with_gt = json.load(f)

ann_dict = {"images": [], "annotations": []}
results = []
gt_id = 0
for item in results_with_gt:
    image_id = item['image_id']
    ann_dict["images"].append({
        "id": image_id,
        "file_name": ""
    })
    gts = item['gt']
    if isinstance(gts, list):
        for gt in gts:
            ann_dict["annotations"].append({
                "image_id": image_id,
                "id": gt_id,
                "caption": gt
            })
            gt_id += 1
    else:
        ann_dict["annotations"].append({
            "image_id": image_id,
            "id": gt_id,
            "caption": gts
        })
        gt_id += 1

    results.append({
        'image_id': image_id,
        'caption': item['response']
    })

# create coco object and coco_result object
coco = COCO()
coco.dataset = ann_dict
coco.createIndex()
coco_result = coco.loadRes(results)

# create coco_eval object by taking coco and coco_result
coco_eval = COCOEvalCap(coco, coco_result)

# evaluate on a subset of images by setting
# coco_eval.params['image_id'] = coco_result.getImgIds()
# please remove this line when evaluating the full validation set
coco_eval.params['image_id'] = coco_result.getImgIds()

# evaluate results
# SPICE will take a few minutes the first time, but speeds up due to caching
coco_eval.evaluate()

# print output evaluation scores
for metric, score in coco_eval.eval.items():
    print(f'{metric}: {score:.3f}')
