import json
json_path = './output/eval/video__video_finetune_1epoch/20241125_092045_results.json'
with open(json_path, 'r') as f:
    data = json.load(f)

result = data['results']
with open('./output/eval/video__video_finetune_1epoch/results.json', 'w') as res:
    res_dict = {}
    for key in result.keys():
        if key == 'mvbench':
            continue
        res_dict[key] = result[key]['mvbench_accuracy,none']
    json.dump(res_dict, res)
