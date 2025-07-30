import json
import argparse


def read_json(path):
    result = json.load(open(path, 'r'))
    return result

def write_json(path, outputs):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=4)
    print('Done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process OCR predictions and merge with raw data')
    parser.add_argument('--predict_path', type=str, default=None, 
                       help='Path to prediction file or directory containing prediction files')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Path to save the output JSON file')
    parser.add_argument('--predict_model', type=str, default='eagle',
                       help='Model name for prediction (default: eagle)')
    
    args = parser.parse_args()
    
    # data_names = ['doc', 'table', 'formula']
    # data_names = ['vqa']
    # data_names = ['vqa', 'doc', 'table', 'formula']
    data_names = ['all_bbox']
    outputs = []
    predict_model = args.predict_model
    
    # Use command line argument for output_path if provided, otherwise use default
    if args.output_path:
        output_path = args.output_path
    else:
        output_path = f'./MultimodalOCR-main/OCRBench_v2/pred_folder/vqa_{predict_model}.json'
    
    for data_name in data_names:
        # raw_path = f'OCRBench_v2/parsing_{data_name}.json'
        raw_path = f'OCRBench_v2/OCRBench_v2.json'
        raw_data = read_json(raw_path)
        
        # Use command line argument for predict_path if provided, otherwise use default logic
        if args.predict_path:
            predict_path = args.predict_path
        else:
            if predict_model == 'onellm':
                predict_path = f'onellm/eval_ocrbench/{data_name}.json'
            else:
                predict_path = f'eagle_ocr/0726/{data_name}.json'
        
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
