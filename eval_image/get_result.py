import json
import argparse


def read_json(path):
    result = json.load(open(path, 'r'))
    return result

def write_json(path, outputs):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=4)
    print('Done')

def process_predictions(data_names, predict_model, predict_path=None, raw_data='OCRBench_v2.json'):
    """
    Process OCR predictions and merge with raw data
    
    Args:
        data_names (list): List of data names to process
        predict_model (str): Model name for prediction
        predict_path (str, optional): Path to prediction file. If None, uses default logic
        raw_data (str): Name of the JSON data file to use for evaluation
    
    Returns:
        list: List of processed outputs with predictions merged
    """
    outputs = []
    
    for data_name in data_names:
        # raw_path = f'OCRBench_v2/parsing_{data_name}.json'
        raw_path = f'OCRBench_v2/{raw_data}'
        raw_data_content = read_json(raw_path)
        
        # Use provided predict_path if available, otherwise use default logic
        if predict_path:
            current_predict_path = predict_path
        else:
            if predict_model == 'onellm':
                current_predict_path = f'onellm/eval_ocrbench/{data_name}.json'
            else:
                current_predict_path = f'eagle_ocr/0726/{data_name}.json'
        
        predict_data = read_json(current_predict_path)
        
        for raw, predict in zip(raw_data_content, predict_data):
            output = raw
            if predict_model == 'onellm':
                output['predict'] = predict['predict']
            else:
                output['predict'] = predict['prediction'][0]
            outputs.append(output)
    
    return outputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process OCR predictions and merge with raw data')
    parser.add_argument('--predict_path', type=str, default=None, 
                       help='Path to prediction file or directory containing prediction files')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Path to save the output JSON file')
    parser.add_argument('--predict_model', type=str, default='eagle',
                       help='Model name for prediction (default: eagle)')
    parser.add_argument('--raw_data', type=str, default='OCRBench_v2.json',
                       help='Name of the JSON data file to use for evaluation (e.g., OCRBench_v2.json)')
    
    args = parser.parse_args()
    
    # data_names = ['doc', 'table', 'formula']
    # data_names = ['vqa']
    # data_names = ['vqa', 'doc', 'table', 'formula']
    data_names = ['all_bbox']
    
    # Use command line argument for output_path if provided, otherwise use default
    if args.output_path:
        output_path = args.output_path
    else:
        output_path = f'./MultimodalOCR-main/OCRBench_v2/pred_folder/vqa_{args.predict_model}.json'
    
    # Process predictions using the extracted function
    outputs = process_predictions(data_names, args.predict_model, args.predict_path, args.raw_data)

    write_json(output_path, outputs)
    print(f'Output saved to {output_path}')
