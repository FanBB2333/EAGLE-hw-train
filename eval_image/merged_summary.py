import os
import pandas as pd
import numpy as np
from pathlib import Path
CURRENT_DIR = Path(__file__).parent
RESULTS_DIR = CURRENT_DIR / "res_folder/images"


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
        df_json = pd.read_json(latest_json)
        # json format as follows
    
    # output to csv
    
        
    
    

if __name__ == "__main__":
    pass