import json


def read_json(path):
    result = json.load(open(path, 'r'))
    return result

def write_json(path, outputs):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=4)
    print('Done')

if __name__ == '__main__':
    # data_names = ['doc', 'table', 'formula']
    # data_names = ['vqa']
    # data_names = ['vqa', 'doc', 'table', 'formula']
    # data_names = ['all_bbox']
    data_names = ['all_new_score']
    outputs = []
    predict_model = 'eagle'
    output_path = f'./MultimodalOCR-main/OCRBench_v2/pred_folder/vqa_{predict_model}.json'
    for data_name in data_names:
        # raw_path = f'OCRBench_v2/parsing_{data_name}.json'
        raw_path = f'OCRBench_v2/OCRBench_v2.json'
        raw_data = read_json(raw_path)
        if predict_model == 'onellm':
            predict_path = f'onellm/eval_ocrbench/{data_name}.json'
        else:
            predict_path = f'eagle_ocr/0725/{data_name}.json'
        predict_data = read_json(predict_path)
        for raw, predict in zip(raw_data, predict_data):
            output = raw
            if predict_model == 'onellm':
                output['predict'] = predict['predict']
            else:
                output['predict'] = predict['prediction'][0]
            outputs.append(output)

    write_json(output_path, outputs)
    print(f'Output saved to {output_path}')
