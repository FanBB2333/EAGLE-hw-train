import os
import pandas as pd
import numpy as np
import json
from pathlib import Path
CURRENT_DIR = Path(__file__).parent
RESULTS_DIR = CURRENT_DIR / "res_folder/images"

onellm_results = {
    'mmlu': 23.17,
    'mme_perception': 1382,
    'docvqa': 0.0514,
    'chartqa': 0.072,
    'textvqa': 0.279,
    'ocrbenchv2': 0.1485
}


def get_model_dirs():
    models_name = [d.name for d in RESULTS_DIR.iterdir() if d.is_dir()]
    # filter the dir names that contains '_'
    models_name = [name for name in models_name if '_' in name]
    return models_name


def convert_model():
    models_name = get_model_dirs()
    df_list = []
    # for each model, get the summary_all_20250801_162145.json format json and find the latest one
    for model_name in models_name:
        model_dir = RESULTS_DIR / model_name
        json_files = list(model_dir.glob("summary_all_*.json"))
        if not json_files:
            continue
        # sort by the date in the filename
        json_files.sort(key=lambda x: x.stem.split('_')[-1])
        latest_json = json_files[-1]
        # read the json file
        with open(latest_json, 'r') as f:
            data = json.load(f)
        
        # Extract scores from the JSON data
        model_scores = {'model_name': model_name}
        
        # Extract MMLU score
        if 'summary_statistics' in data and 'eval_mmlu' in data['summary_statistics']:
            model_scores['mmlu'] = data['summary_statistics']['eval_mmlu'].get('overall_accuracy', None)
        else:
            model_scores['mmlu'] = None
            
        # Extract TextVQA, DocVQA, ChartQA accuracy
        if 'summary_statistics' in data and 'eval_docvqa_textvqa_chartqa' in data['summary_statistics']:
            stats = data['summary_statistics']['eval_docvqa_textvqa_chartqa']
            model_scores['textvqa_accuracy'] = stats.get('textvqa_accuracy', None)
            model_scores['docvqa_accuracy'] = stats.get('docvqa_accuracy', None)
            model_scores['chartqa_accuracy'] = stats.get('chartqa_accuracy', None)
        else:
            model_scores['textvqa_accuracy'] = None
            model_scores['docvqa_accuracy'] = None
            model_scores['chartqa_accuracy'] = None
            
        # Extract perception_score and english_overall from detailed_results
        model_scores['perception_score'] = None
        model_scores['english_overall'] = None
        
        if 'detailed_results' in data:
            for result in data['detailed_results']:
                # Extract perception_score from MME results
                if result.get('script') == 'eval_mme.py' and 'results' in result:
                    mme_results = result['results'].get('mme', {})
                    model_scores['perception_score'] = mme_results.get('perception_score', None)
                
                # Extract english_overall from OCRBench results
                if result.get('script') == 'eval_ocrbenchv2.py' and 'results' in result:
                    ocr_results = result['results'].get('ocrbenchv2', {})
                    parsed_scores = ocr_results.get('parsed_scores', {})
                    overall_scores = parsed_scores.get('overall_scores', {})
                    model_scores['english_overall'] = overall_scores.get('english_overall', None)
        
        df_list.append(model_scores)
    
    # Convert to DataFrame and output to CSV
    if df_list:
        df = pd.DataFrame(df_list)
        output_file = CURRENT_DIR / "merged_results.csv"
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        return df
    else:
        print("No data found to process")
        return None


if __name__ == "__main__":
    df = convert_model()
    if df is not None:
        print("\nSummary of extracted scores:")
        print(df.to_string(index=False))