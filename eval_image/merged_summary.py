import os
import pandas as pd
import numpy as np
import json
from pathlib import Path
CURRENT_DIR = Path(__file__).parent
RESULTS_DIR = CURRENT_DIR / "res_folder/images"

onellm_results = {
    'mmlu': 0.2317,
    'mme_perception': 1382,
    'docvqa': 0.0514,
    'chartqa': 0.072,
    'textvqa': 0.279,
    'ocrbenchv2': 0.1485
}


def format_score_with_ratio(score, baseline_score):
    """Format score with ratio to baseline in format 'score[ratio%]'"""
    if score is None or baseline_score is None or baseline_score == 0:
        return score
    
    ratio = score / baseline_score
    ratio_percent = ratio * 100
    return f"{score:.3f}[{ratio_percent:.2f}%]"


def format_score_with_color(score, baseline_score, column_name):
    """Format score with color coding based on ratio for terminal display"""
    if score is None or baseline_score is None or baseline_score == 0:
        return str(score)
    
    ratio = score / baseline_score
    ratio_percent = ratio * 100
    formatted_score = f"{score:.3f}[{ratio_percent:.2f}%]"
    
    # Special coloring for mme_perception column: Red for >=100%, no color otherwise
    if column_name == 'mme_perception':
        if ratio_percent >= 100:
            return f"\033[91m{formatted_score}\033[0m"  # Red
        else:
            return formatted_score
    
    # Color codes for other columns: Red for >150%, Yellow for 120%-150%, no color for <120%
    if ratio_percent > 150:
        return f"\033[91m{formatted_score}\033[0m"  # Red
    elif ratio_percent >= 120:
        return f"\033[93m{formatted_score}\033[0m"  # Yellow
    else:
        return formatted_score


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
            raw_mmlu = data['summary_statistics']['eval_mmlu'].get('overall_accuracy', None)
            model_scores['mmlu'] = format_score_with_ratio(raw_mmlu, onellm_results['mmlu'])
        else:
            model_scores['mmlu'] = None
            
        # Extract TextVQA, DocVQA, ChartQA accuracy
        if 'summary_statistics' in data and 'eval_docvqa_textvqa_chartqa' in data['summary_statistics']:
            stats = data['summary_statistics']['eval_docvqa_textvqa_chartqa']
            raw_textvqa = stats.get('textvqa_accuracy', None)
            raw_docvqa = stats.get('docvqa_accuracy', None)
            raw_chartqa = stats.get('chartqa_accuracy', None)
            
            model_scores['textvqa'] = format_score_with_ratio(raw_textvqa, onellm_results['textvqa'])
            model_scores['docvqa'] = format_score_with_ratio(raw_docvqa, onellm_results['docvqa'])
            model_scores['chartqa'] = format_score_with_ratio(raw_chartqa, onellm_results['chartqa'])
        else:
            model_scores['textvqa'] = None
            model_scores['docvqa'] = None
            model_scores['chartqa'] = None
            
        # Extract mme_perception and ocrbenchv2 from detailed_results
        model_scores['mme_perception'] = None
        model_scores['ocrbenchv2'] = None
        
        if 'detailed_results' in data:
            for result in data['detailed_results']:
                # Extract mme_perception from MME results
                if result.get('script') == 'eval_mme.py' and 'results' in result:
                    mme_results = result['results'].get('mme', {})
                    raw_perception = mme_results.get('perception_score', None)
                    model_scores['mme_perception'] = format_score_with_ratio(raw_perception, onellm_results['mme_perception'])
                
                # Extract ocrbenchv2 from OCRBench results
                if result.get('script') == 'eval_ocrbenchv2.py' and 'results' in result:
                    ocr_results = result['results'].get('ocrbenchv2', {})
                    parsed_scores = ocr_results.get('parsed_scores', {})
                    overall_scores = parsed_scores.get('overall_scores', {})
                    raw_english = overall_scores.get('english_overall', None)
                    model_scores['ocrbenchv2'] = format_score_with_ratio(raw_english, onellm_results['ocrbenchv2'])

        df_list.append(model_scores)
    
    # Convert to DataFrame and output to CSV
    if df_list:
        df = pd.DataFrame(df_list)
        
        # Add onellm_results as the first row
        onellm_row = {
            'model_name': 'onellm',
            'mmlu': f"{onellm_results['mmlu']:.3f}",
            'textvqa': f"{onellm_results['textvqa']:.3f}",
            'docvqa': f"{onellm_results['docvqa']:.3f}",
            'chartqa': f"{onellm_results['chartqa']:.3f}",
            'mme_perception': onellm_results['mme_perception'],  # Keep as is since it's already a large number
            'ocrbenchv2': f"{onellm_results['ocrbenchv2']:.3f}"
        }
        
        # Insert onellm_results as the first row
        df = pd.concat([pd.DataFrame([onellm_row]), df], ignore_index=True)
        
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
        
        # Create a colored version for terminal display
        df_display = df.copy()
        
        # First, let's get the maximum width for each column for proper alignment
        col_widths = {}
        for col in df.columns:
            col_widths[col] = max(len(col), max(len(str(val)) for val in df[col] if val is not None))
        
        # Apply color formatting to all columns except model_name
        for index, row in df.iterrows():
            if index == 0:  # Skip the baseline row
                continue
                
            for col in df.columns:
                if col == 'model_name':
                    continue
                    
                # Extract the original score from the formatted string
                value = row[col]
                if value is None or value == 'None':
                    continue
                    
                try:
                    # Parse the score from formatted string like "0.235[101.50%]"
                    if '[' in str(value) and ']' in str(value):
                        score_str = str(value).split('[')[0]
                        score = float(score_str)
                        
                        # Get baseline score for this column
                        baseline_score = None
                        if col == 'mmlu':
                            baseline_score = onellm_results['mmlu']
                        elif col == 'textvqa':
                            baseline_score = onellm_results['textvqa']
                        elif col == 'docvqa':
                            baseline_score = onellm_results['docvqa']
                        elif col == 'chartqa':
                            baseline_score = onellm_results['chartqa']
                        elif col == 'mme_perception':
                            baseline_score = onellm_results['mme_perception']
                        elif col == 'ocrbenchv2':
                            baseline_score = onellm_results['ocrbenchv2']
                        
                        if baseline_score is not None:
                            colored_value = format_score_with_color(score, baseline_score, col)
                            df_display.at[index, col] = colored_value
                except (ValueError, IndexError):
                    # If parsing fails, keep original value
                    pass
        
        # Custom formatting to handle ANSI color codes properly
        def print_aligned_table(df):
            # Calculate the actual display width of strings (ignoring ANSI codes)
            def display_len(s):
                # Remove ANSI escape sequences for length calculation
                import re
                ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
                return len(ansi_escape.sub('', str(s)))
            
            # Calculate column widths based on visible content
            widths = {}
            for col in df.columns:
                widths[col] = max(len(col), max(display_len(str(val)) for val in df[col] if val is not None))
            
            # Print header
            header = "  ".join(col.ljust(widths[col]) for col in df.columns)
            print(header)
            
            # Print rows
            for _, row in df.iterrows():
                formatted_row = []
                for col in df.columns:
                    val = str(row[col]) if row[col] is not None else "None"
                    # Calculate padding needed (accounting for ANSI codes)
                    padding = widths[col] - display_len(val)
                    formatted_row.append(val + " " * padding)
                print("  ".join(formatted_row))
        
        print_aligned_table(df_display)